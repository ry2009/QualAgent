import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# AndroidEnv imports (this would be the actual android_world library)
try:
    from android_env import loader
    from android_env.components import action_type as action_type_lib
    from android_env.components import coordinator as coordinator_lib
    ANDROID_ENV_AVAILABLE = True
except ImportError:
    ANDROID_ENV_AVAILABLE = False
    logging.warning("android_env not available. Using mock implementation.")

from ..models.ui_state import UIState, UIElement, ElementType
from ..models.result import ExecutionResult, ResultStatus

@dataclass
class AndroidAction:
    """Represents an action to be executed in the Android environment"""
    action_type: str  # touch, type, scroll, key, wait
    coordinates: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    element_id: Optional[str] = None
    direction: Optional[str] = None  # up, down, left, right
    distance: Optional[int] = None
    key_code: Optional[int] = None
    duration_ms: Optional[int] = None

class AndroidWorldIntegration:
    """Integration layer for Android World environment"""
    
    def __init__(self, 
                 task_name: str = "default",
                 avd_name: str = "AndroidWorldAvd",
                 enable_screenshots: bool = True,
                 screenshot_quality: int = 80):
        
        self.task_name = task_name
        self.avd_name = avd_name
        self.enable_screenshots = enable_screenshots
        self.screenshot_quality = screenshot_quality
        self.logger = logging.getLogger("android_integration")
        
        # Environment state
        self.env = None
        self.is_connected = False
        self.current_observation = None
        self.current_ui_state: Optional[UIState] = None
        
        # Performance tracking
        self.action_count = 0
        self.successful_actions = 0
        self.total_execution_time = 0
        
        # Initialize environment
        if ANDROID_ENV_AVAILABLE:
            self._initialize_real_env()
        else:
            self._initialize_mock_env()
    
    def _initialize_real_env(self):
        """Initialize real Android environment"""
        try:
            self.logger.info(f"Initializing Android environment for task: {self.task_name}")
            
            # Load the environment
            self.env = loader.load(
                task_name=self.task_name,
                avd_name=self.avd_name
            )
            
            self.logger.info("Android environment initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Android environment: {e}")
            self._initialize_mock_env()
    
    def _initialize_mock_env(self):
        """Initialize mock environment for testing"""
        self.logger.info("Using mock Android environment")
        self.env = MockAndroidEnv()
    
    async def connect(self) -> bool:
        """Connect to the Android environment"""
        try:
            if not self.env:
                raise RuntimeError("Environment not initialized")
            
            # Reset environment
            self.current_observation = self.env.reset()
            self.is_connected = True
            
            # Get initial UI state
            await self.update_ui_state()
            
            self.logger.info("Connected to Android environment")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Android environment: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the Android environment"""
        try:
            if self.env and hasattr(self.env, 'close'):
                self.env.close()
            
            self.is_connected = False
            self.current_observation = None
            self.current_ui_state = None
            
            self.logger.info("Disconnected from Android environment")
            
        except Exception as e:
            self.logger.error(f"Error during disconnection: {e}")
    
    async def execute_action(self, action: AndroidAction) -> ExecutionResult:
        """Execute an action in the Android environment"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Android environment")
        
        start_time = time.time()
        result = ExecutionResult(
            action_type=action.action_type,
            timestamp=time.time()
        )
        
        # Capture screenshot before action
        if self.enable_screenshots:
            result.screenshot_before = await self.capture_screenshot()
            result.ui_elements_before = self._get_current_ui_elements()
        
        try:
            # Execute the action based on type
            if action.action_type == "touch":
                await self._execute_touch(action, result)
            elif action.action_type == "type":
                await self._execute_type(action, result)
            elif action.action_type == "scroll":
                await self._execute_scroll(action, result)
            elif action.action_type == "key":
                await self._execute_key(action, result)
            elif action.action_type == "wait":
                await self._execute_wait(action, result)
            else:
                raise ValueError(f"Unsupported action type: {action.action_type}")
            
            # Update UI state after action
            await self.update_ui_state()
            
            # Capture screenshot after action
            if self.enable_screenshots:
                result.screenshot_after = await self.capture_screenshot()
                result.ui_elements_after = self._get_current_ui_elements()
            
            # Calculate execution time
            execution_time = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time
            
            # Mark as successful
            result.mark_success()
            self.successful_actions += 1
            
            self.logger.info(f"Action executed successfully: {action.action_type} "
                           f"({execution_time}ms)")
            
        except Exception as e:
            result.mark_failure(str(e), "execution_error")
            self.logger.error(f"Action execution failed: {e}")
        
        finally:
            self.action_count += 1
            self.total_execution_time += result.execution_time_ms
        
        return result
    
    async def _execute_touch(self, action: AndroidAction, result: ExecutionResult):
        """Execute touch action"""
        if not action.coordinates:
            raise ValueError("Touch action requires coordinates")
        
        x, y = action.coordinates
        result.actual_coordinates = (x, y)
        
        # Convert to Android environment action format
        android_action = {
            "action_type": action_type_lib.ActionType.TOUCH,
            "touch_point": np.array([x, y], dtype=np.float32)
        }
        
        # Execute action
        timestep = self.env.step(android_action)
        self.current_observation = timestep.observation
        
        self.logger.debug(f"Touch executed at ({x}, {y})")
    
    async def _execute_type(self, action: AndroidAction, result: ExecutionResult):
        """Execute type action"""
        if not action.text:
            raise ValueError("Type action requires text")
        
        result.text_entered = action.text
        
        # Type each character
        for char in action.text:
            android_action = {
                "action_type": action_type_lib.ActionType.TYPE,
                "text": char
            }
            
            timestep = self.env.step(android_action)
            self.current_observation = timestep.observation
            
            # Small delay between characters
            await asyncio.sleep(0.05)
        
        self.logger.debug(f"Text typed: {action.text}")
    
    async def _execute_scroll(self, action: AndroidAction, result: ExecutionResult):
        """Execute scroll action"""
        if not action.coordinates or not action.direction:
            raise ValueError("Scroll action requires coordinates and direction")
        
        x, y = action.coordinates
        direction = action.direction.lower()
        distance = action.distance or 300
        
        # Calculate end coordinates based on direction
        if direction == "up":
            end_x, end_y = x, y - distance
        elif direction == "down":
            end_x, end_y = x, y + distance
        elif direction == "left":
            end_x, end_y = x - distance, y
        elif direction == "right":
            end_x, end_y = x + distance, y
        else:
            raise ValueError(f"Invalid scroll direction: {direction}")
        
        result.actual_coordinates = (x, y)
        result.scroll_distance = (end_x - x, end_y - y)
        
        # Execute scroll action
        android_action = {
            "action_type": action_type_lib.ActionType.SCROLL,
            "start_point": np.array([x, y], dtype=np.float32),
            "end_point": np.array([end_x, end_y], dtype=np.float32)
        }
        
        timestep = self.env.step(android_action)
        self.current_observation = timestep.observation
        
        self.logger.debug(f"Scroll executed: {direction} from ({x}, {y})")
    
    async def _execute_key(self, action: AndroidAction, result: ExecutionResult):
        """Execute key press action"""
        if not action.key_code:
            raise ValueError("Key action requires key_code")
        
        android_action = {
            "action_type": action_type_lib.ActionType.KEY,
            "key_code": action.key_code
        }
        
        timestep = self.env.step(android_action)
        self.current_observation = timestep.observation
        
        self.logger.debug(f"Key pressed: {action.key_code}")
    
    async def _execute_wait(self, action: AndroidAction, result: ExecutionResult):
        """Execute wait action"""
        duration = action.duration_ms or 1000
        await asyncio.sleep(duration / 1000.0)
        
        self.logger.debug(f"Waited for {duration}ms")
    
    async def update_ui_state(self):
        """Update the current UI state from the environment"""
        try:
            if not self.current_observation:
                return
            
            # Extract UI hierarchy from observation
            ui_elements = self._parse_ui_hierarchy()
            
            # Create UI state
            self.current_ui_state = UIState(
                elements=ui_elements,
                screen_width=self.current_observation.get('screen_width', 1080),
                screen_height=self.current_observation.get('screen_height', 1920),
                current_app=self.current_observation.get('foreground_activity', {}).get('package'),
                current_activity=self.current_observation.get('foreground_activity', {}).get('activity')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update UI state: {e}")
    
    def _parse_ui_hierarchy(self) -> List[UIElement]:
        """Parse UI hierarchy from environment observation"""
        elements = []
        
        try:
            # Get view hierarchy from observation
            view_hierarchy = self.current_observation.get('view_hierarchy', [])
            
            for i, element_data in enumerate(view_hierarchy):
                element = UIElement(
                    resource_id=element_data.get('resource_id'),
                    class_name=element_data.get('class'),
                    text=element_data.get('text'),
                    content_description=element_data.get('content_description'),
                    bounds=self._parse_bounds(element_data.get('bounds')),
                    is_clickable=element_data.get('clickable', False),
                    is_long_clickable=element_data.get('long_clickable', False),
                    is_scrollable=element_data.get('scrollable', False),
                    is_editable=element_data.get('editable', False),
                    is_enabled=element_data.get('enabled', True),
                    is_selected=element_data.get('selected', False),
                    is_checked=element_data.get('checked', False),
                    is_focused=element_data.get('focused', False),
                    depth=element_data.get('depth', 0),
                    index_in_parent=i
                )
                
                elements.append(element)
        
        except Exception as e:
            self.logger.error(f"Error parsing UI hierarchy: {e}")
        
        return elements
    
    def _parse_bounds(self, bounds_str: str) -> Optional[Tuple[int, int, int, int]]:
        """Parse bounds string to tuple"""
        try:
            if not bounds_str:
                return None
            
            # Expected format: "[left,top][right,bottom]"
            bounds_str = bounds_str.replace('[', '').replace(']', '')
            parts = bounds_str.split(',')
            
            if len(parts) == 4:
                return tuple(int(x) for x in parts)
        
        except Exception:
            pass
        
        return None
    
    def _get_current_ui_elements(self) -> List[Dict[str, Any]]:
        """Get current UI elements as dictionaries"""
        if not self.current_ui_state:
            return []
        
        return [elem.to_dict() for elem in self.current_ui_state.elements]
    
    async def capture_screenshot(self) -> Optional[bytes]:
        """Capture screenshot of current screen"""
        try:
            if not self.current_observation:
                return None
            
            # Get screenshot from observation
            screenshot_data = self.current_observation.get('pixels')
            
            if screenshot_data is not None:
                # Convert to PIL Image
                if isinstance(screenshot_data, np.ndarray):
                    image = Image.fromarray(screenshot_data)
                else:
                    image = Image.open(BytesIO(screenshot_data))
                
                # Compress image
                output = BytesIO()
                image.save(output, format='JPEG', quality=self.screenshot_quality)
                return output.getvalue()
        
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {e}")
        
        return None
    
    def get_current_ui_state(self) -> Optional[UIState]:
        """Get the current UI state"""
        return self.current_ui_state
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        success_rate = (
            self.successful_actions / self.action_count 
            if self.action_count > 0 else 0.0
        )
        
        avg_execution_time = (
            self.total_execution_time / self.action_count
            if self.action_count > 0 else 0.0
        )
        
        return {
            "is_connected": self.is_connected,
            "task_name": self.task_name,
            "total_actions": self.action_count,
            "successful_actions": self.successful_actions,
            "success_rate": round(success_rate * 100, 2),
            "average_execution_time_ms": round(avg_execution_time, 2),
            "current_app": self.current_ui_state.current_app if self.current_ui_state else None,
            "current_activity": self.current_ui_state.current_activity if self.current_ui_state else None
        }
    
    async def wait_for_ui_stable(self, timeout_seconds: int = 10) -> bool:
        """Wait for UI to stabilize (no changes for a period)"""
        stable_duration = 2.0  # seconds
        check_interval = 0.5   # seconds
        
        last_hash = None
        stable_start = None
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            await self.update_ui_state()
            
            current_hash = self.current_ui_state.hierarchy_hash if self.current_ui_state else None
            
            if current_hash == last_hash:
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= stable_duration:
                    return True
            else:
                stable_start = None
                last_hash = current_hash
            
            await asyncio.sleep(check_interval)
        
        return False

class MockAndroidEnv:
    """Mock Android environment for testing"""
    
    def __init__(self):
        self.observation_count = 0
    
    def reset(self):
        """Reset environment"""
        self.observation_count = 0
        return self._create_mock_observation()
    
    def step(self, action):
        """Execute step"""
        self.observation_count += 1
        
        # Create mock timestep
        class MockTimestep:
            def __init__(self, observation):
                self.observation = observation
        
        return MockTimestep(self._create_mock_observation())
    
    def _create_mock_observation(self):
        """Create mock observation"""
        return {
            'pixels': np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8),
            'view_hierarchy': [
                {
                    'resource_id': 'com.example.app:id/button1',
                    'class': 'android.widget.Button',
                    'text': 'Click Me',
                    'bounds': '[100,200][300,250]',
                    'clickable': True,
                    'enabled': True
                },
                {
                    'resource_id': 'com.example.app:id/text1',
                    'class': 'android.widget.TextView',
                    'text': 'Hello World',
                    'bounds': '[100,100][300,150]',
                    'clickable': False,
                    'enabled': True
                }
            ],
            'foreground_activity': {
                'package': 'com.example.app',
                'activity': 'MainActivity'
            },
            'screen_width': 1080,
            'screen_height': 1920
        }
    
    def close(self):
        """Close environment"""
        pass 