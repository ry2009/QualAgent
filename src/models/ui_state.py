from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import hashlib

class ElementType(Enum):
    BUTTON = "button"
    TEXT_VIEW = "text_view"
    EDIT_TEXT = "edit_text"
    IMAGE_VIEW = "image_view"
    LAYOUT = "layout"
    LIST_VIEW = "list_view"
    SCROLL_VIEW = "scroll_view"
    WEB_VIEW = "web_view"
    DIALOG = "dialog"
    MENU = "menu"
    TAB = "tab"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    SWITCH = "switch"
    PROGRESS_BAR = "progress_bar"
    UNKNOWN = "unknown"

class ElementState(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    SELECTED = "selected"
    UNSELECTED = "unselected"
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    FOCUSED = "focused"
    UNFOCUSED = "unfocused"
    VISIBLE = "visible"
    INVISIBLE = "invisible"
    GONE = "gone"

@dataclass
class UIElement:
    """Represents a single UI element in the Android interface"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Element identification
    resource_id: Optional[str] = None
    class_name: Optional[str] = None
    content_description: Optional[str] = None
    text: Optional[str] = None
    hint_text: Optional[str] = None
    
    # Element type and properties
    element_type: ElementType = ElementType.UNKNOWN
    package_name: Optional[str] = None
    
    # Position and size
    bounds: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    
    # Element states
    is_clickable: bool = False
    is_long_clickable: bool = False
    is_scrollable: bool = False
    is_editable: bool = False
    is_checkable: bool = False
    is_enabled: bool = True
    is_selected: bool = False
    is_checked: bool = False
    is_focused: bool = False
    is_password: bool = False
    
    # Hierarchy information
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    index_in_parent: int = 0
    
    # Additional metadata
    xpath: Optional[str] = None
    screenshot_region: Optional[bytes] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived properties after initialization"""
        if self.bounds:
            left, top, right, bottom = self.bounds
            self.center_x = (left + right) // 2
            self.center_y = (top + bottom) // 2
            self.width = right - left
            self.height = bottom - top
        
        # Determine element type from class name if not set
        if self.element_type == ElementType.UNKNOWN and self.class_name:
            self.element_type = self._infer_element_type()
    
    def _infer_element_type(self) -> ElementType:
        """Infer element type from class name"""
        if not self.class_name:
            return ElementType.UNKNOWN
            
        class_lower = self.class_name.lower()
        
        if "button" in class_lower:
            return ElementType.BUTTON
        elif "textview" in class_lower:
            return ElementType.TEXT_VIEW
        elif "edittext" in class_lower:
            return ElementType.EDIT_TEXT
        elif "imageview" in class_lower:
            return ElementType.IMAGE_VIEW
        elif "listview" in class_lower:
            return ElementType.LIST_VIEW
        elif "scrollview" in class_lower:
            return ElementType.SCROLL_VIEW
        elif "webview" in class_lower:
            return ElementType.WEB_VIEW
        elif "layout" in class_lower:
            return ElementType.LAYOUT
        elif "checkbox" in class_lower:
            return ElementType.CHECKBOX
        elif "radiobutton" in class_lower:
            return ElementType.RADIO_BUTTON
        elif "switch" in class_lower:
            return ElementType.SWITCH
        elif "progress" in class_lower:
            return ElementType.PROGRESS_BAR
        else:
            return ElementType.UNKNOWN
    
    def get_center_coordinates(self) -> Tuple[int, int]:
        """Get the center coordinates of the element"""
        if self.center_x is not None and self.center_y is not None:
            return (self.center_x, self.center_y)
        elif self.bounds:
            left, top, right, bottom = self.bounds
            return ((left + right) // 2, (top + bottom) // 2)
        else:
            raise ValueError("Element bounds not available")
    
    def is_interactable(self) -> bool:
        """Check if the element can be interacted with"""
        return (self.is_enabled and 
                (self.is_clickable or self.is_long_clickable or 
                 self.is_scrollable or self.is_editable))
    
    def get_text_content(self) -> str:
        """Get the most relevant text content of the element"""
        if self.text:
            return self.text
        elif self.content_description:
            return self.content_description
        elif self.hint_text:
            return self.hint_text
        else:
            return ""
    
    def matches_selector(self, selector: Dict[str, Any]) -> bool:
        """Check if element matches the given selector criteria"""
        for key, value in selector.items():
            if key == "resource_id" and self.resource_id != value:
                return False
            elif key == "class_name" and self.class_name != value:
                return False
            elif key == "text" and self.text != value:
                return False
            elif key == "content_description" and self.content_description != value:
                return False
            elif key == "element_type" and self.element_type.value != value:
                return False
            elif key == "is_clickable" and self.is_clickable != value:
                return False
            elif key == "is_enabled" and self.is_enabled != value:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert element to dictionary for serialization"""
        return {
            "id": self.id,
            "resource_id": self.resource_id,
            "class_name": self.class_name,
            "text": self.text,
            "content_description": self.content_description,
            "element_type": self.element_type.value,
            "bounds": self.bounds,
            "center": (self.center_x, self.center_y) if self.center_x and self.center_y else None,
            "size": (self.width, self.height) if self.width and self.height else None,
            "is_clickable": self.is_clickable,
            "is_enabled": self.is_enabled,
            "is_editable": self.is_editable,
            "depth": self.depth,
            "xpath": self.xpath
        }

@dataclass
class UIState:
    """Represents the complete state of the Android UI at a specific moment"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # UI elements
    elements: List[UIElement] = field(default_factory=list)
    root_elements: List[str] = field(default_factory=list)
    
    # Screen information
    screen_width: int = 0
    screen_height: int = 0
    orientation: str = "portrait"  # portrait, landscape
    density: float = 1.0
    
    # App context
    current_app: Optional[str] = None
    current_activity: Optional[str] = None
    current_package: Optional[str] = None
    
    # UI hierarchy hash for change detection
    hierarchy_hash: Optional[str] = None
    
    # Screenshot
    screenshot: Optional[bytes] = None
    
    # Navigation context
    can_go_back: bool = False
    can_go_forward: bool = False
    is_keyboard_shown: bool = False
    
    def __post_init__(self):
        """Calculate derived properties after initialization"""
        self._build_hierarchy()
        self.hierarchy_hash = self._calculate_hierarchy_hash()
    
    def _build_hierarchy(self):
        """Build parent-child relationships and find root elements"""
        element_map = {elem.id: elem for elem in self.elements}
        
        # Clear existing hierarchy info
        self.root_elements = []
        
        for element in self.elements:
            # Find children
            element.children_ids = [
                child.id for child in self.elements 
                if child.parent_id == element.id
            ]
            
            # Identify root elements (no parent)
            if not element.parent_id:
                self.root_elements.append(element.id)
    
    def _calculate_hierarchy_hash(self) -> str:
        """Calculate a hash of the UI hierarchy for change detection"""
        hierarchy_data = []
        for element in sorted(self.elements, key=lambda x: x.id):
            element_data = (
                element.resource_id or "",
                element.class_name or "",
                element.text or "",
                element.bounds or (0, 0, 0, 0),
                element.is_enabled,
                element.is_clickable
            )
            hierarchy_data.append(str(element_data))
        
        hierarchy_string = "|".join(hierarchy_data)
        return hashlib.md5(hierarchy_string.encode()).hexdigest()
    
    def add_element(self, element: UIElement):
        """Add a UI element to the state"""
        self.elements.append(element)
        self._build_hierarchy()
        self.hierarchy_hash = self._calculate_hierarchy_hash()
    
    def find_elements(self, **criteria) -> List[UIElement]:
        """Find elements matching the given criteria"""
        matching_elements = []
        for element in self.elements:
            if element.matches_selector(criteria):
                matching_elements.append(element)
        return matching_elements
    
    def find_element_by_id(self, element_id: str) -> Optional[UIElement]:
        """Find element by its unique ID"""
        for element in self.elements:
            if element.id == element_id:
                return element
        return None
    
    def find_element_by_resource_id(self, resource_id: str) -> Optional[UIElement]:
        """Find element by its Android resource ID"""
        for element in self.elements:
            if element.resource_id == resource_id:
                return element
        return None
    
    def find_clickable_elements(self) -> List[UIElement]:
        """Find all clickable elements"""
        return [elem for elem in self.elements if elem.is_clickable and elem.is_enabled]
    
    def find_editable_elements(self) -> List[UIElement]:
        """Find all editable text elements"""
        return [elem for elem in self.elements if elem.is_editable and elem.is_enabled]
    
    def find_elements_by_text(self, text: str, exact_match: bool = True) -> List[UIElement]:
        """Find elements containing specific text"""
        matching_elements = []
        for element in self.elements:
            element_text = element.get_text_content()
            if exact_match and element_text == text:
                matching_elements.append(element)
            elif not exact_match and text.lower() in element_text.lower():
                matching_elements.append(element)
        return matching_elements
    
    def get_element_at_coordinates(self, x: int, y: int) -> Optional[UIElement]:
        """Find the topmost element at the given coordinates"""
        candidates = []
        for element in self.elements:
            if element.bounds:
                left, top, right, bottom = element.bounds
                if left <= x <= right and top <= y <= bottom:
                    candidates.append(element)
        
        # Return the element with the highest depth (most specific)
        if candidates:
            return max(candidates, key=lambda e: e.depth)
        return None
    
    def has_changed_significantly(self, other_state: 'UIState') -> bool:
        """Check if this state has changed significantly from another state"""
        if not other_state:
            return True
        
        # Compare hierarchy hashes
        if self.hierarchy_hash != other_state.hierarchy_hash:
            return True
        
        # Compare app context
        if (self.current_activity != other_state.current_activity or
            self.current_package != other_state.current_package):
            return True
        
        return False
    
    def get_interactable_elements_summary(self) -> Dict[str, Any]:
        """Get a summary of interactable elements for agent decision making"""
        clickable = self.find_clickable_elements()
        editable = self.find_editable_elements()
        
        summary = {
            "total_elements": len(self.elements),
            "clickable_count": len(clickable),
            "editable_count": len(editable),
            "current_app": self.current_app,
            "current_activity": self.current_activity,
            "keyboard_shown": self.is_keyboard_shown,
            "clickable_elements": [
                {
                    "id": elem.id,
                    "type": elem.element_type.value,
                    "text": elem.get_text_content()[:50],  # Truncate long text
                    "resource_id": elem.resource_id,
                    "center": elem.get_center_coordinates(),
                    "bounds": elem.bounds
                }
                for elem in clickable[:10]  # Limit to first 10
            ],
            "editable_elements": [
                {
                    "id": elem.id,
                    "text": elem.get_text_content()[:50],
                    "hint": elem.hint_text,
                    "resource_id": elem.resource_id,
                    "center": elem.get_center_coordinates()
                }
                for elem in editable[:5]  # Limit to first 5
            ]
        }
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert UI state to dictionary for serialization"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "screen_size": (self.screen_width, self.screen_height),
            "orientation": self.orientation,
            "current_app": self.current_app,
            "current_activity": self.current_activity,
            "current_package": self.current_package,
            "hierarchy_hash": self.hierarchy_hash,
            "elements_count": len(self.elements),
            "root_elements_count": len(self.root_elements),
            "can_go_back": self.can_go_back,
            "is_keyboard_shown": self.is_keyboard_shown,
            "elements": [elem.to_dict() for elem in self.elements]
        } 