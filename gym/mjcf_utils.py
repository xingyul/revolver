# utility functions for manipulating MJCF XML models

import xml.etree.ElementTree as ET
import numpy as np
from copy import deepcopy


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"

    Args:
        array (n-array): Array to convert to a string

    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def convert_to_string(inp):
    """
    Converts any type of {bool, int, float, list, tuple, array, string, np.str_} into an mujoco-xml compatible string.
        Note that an input string / np.str_ results in a no-op action.

    Args:
        inp: Input to convert to string

    Returns:
        str: String equivalent of @inp
    """
    if type(inp) in {list, tuple, np.ndarray}:
        return array_to_string(inp)
    elif type(inp) in {int, float, bool}:
        return str(inp).lower()
    elif type(inp) in {str, np.str_}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def new_element(tag, name, **kwargs):
    """
    Creates a new @tag element with attributes specified by @**kwargs.

    Args:
        tag (str): Type of element to create
        name (None or str): Name for this element. Should only be None for elements that do not have an explicit
            name attribute (e.g.: inertial elements)
        **kwargs: Specified attributes for the new joint

    Returns:
        ET.Element: new specified xml element
    """
    # Name will be set if it's not None
    if name is not None:
        kwargs["name"] = name
    # Loop through all attributes and pop any that are None, otherwise convert them to strings
    for k, v in kwargs.copy().items():
        if v is None:
            kwargs.pop(k)
        else:
            kwargs[k] = convert_to_string(v)
    element = ET.Element(tag, attrib=kwargs)
    return element


def new_joint(name, **kwargs):
    """
    Creates a joint tag with attributes specified by @**kwargs.

    Args:
        name (str): Name for this joint
        **kwargs: Specified attributes for the new joint

    Returns:
        ET.Element: new joint xml element
    """
    return new_element(tag="joint", name=name, **kwargs)


def new_actuator(name, joint, act_type="actuator", **kwargs):
    """
    Creates an actuator tag with attributes specified by @**kwargs.

    Args:
        name (str): Name for this actuator
        joint (str): type of actuator transmission.
            see all types here: http://mujoco.org/book/modeling.html#actuator
        act_type (str): actuator type. Defaults to "actuator"
        **kwargs: Any additional specified attributes for the new joint

    Returns:
        ET.Element: new actuator xml element
    """
    element = new_element(tag=act_type, name=name, **kwargs)
    element.set("joint", joint)
    return element


def new_geom(name, type, size, pos=(0, 0, 0), group=0, **kwargs):
    """
    Creates a geom element with attributes specified by @**kwargs.

    NOTE: With the exception of @geom_type, @size, and @pos, if any arg is set to
        None, the value will automatically be popped before passing the values
        to create the appropriate XML

    Args:
        name (str): Name for this geom
        type (str): type of the geom.
            see all types here: http://mujoco.org/book/modeling.html#geom
        size (n-array of float): geom size parameters.
        pos (3-array): (x,y,z) 3d position of the site.
        group (int): the integrer group that the geom belongs to. useful for
            separating visual and physical elements.
        **kwargs: Any additional specified attributes for the new geom

    Returns:
        ET.Element: new geom xml element
    """
    kwargs["type"] = type
    kwargs["size"] = size
    kwargs["pos"] = pos
    kwargs["group"] = group if group is not None else None
    return new_element(tag="geom", name=name, **kwargs)


def new_body(name, pos=(0, 0, 0), **kwargs):
    """
    Creates a body element with attributes specified by @**kwargs.

    Args:
        name (str): Name for this body
        pos (3-array): (x,y,z) 3d position of the body frame.
        **kwargs: Any additional specified attributes for the new body

    Returns:
        ET.Element: new body xml element
    """
    kwargs["pos"] = pos
    return new_element(tag="body", name=name, **kwargs)


def new_inertial(pos=(0, 0, 0), mass=None, **kwargs):
    """
    Creates a inertial element with attributes specified by @**kwargs.

    Args:
        pos (3-array): (x,y,z) 3d position of the inertial frame.
        mass (float): The mass of inertial
        **kwargs: Any additional specified attributes for the new inertial element

    Returns:
        ET.Element: new inertial xml element
    """
    kwargs["mass"] = mass if mass is not None else None
    kwargs["pos"] = pos
    return new_element(tag="inertial", name=None, **kwargs)


def find_elements(root, tags, attribs=None, return_first=True):
    """
    Find all element(s) matching the requested @tag and @attributes. If @return_first is True, then will return the
    first element found matching the criteria specified. Otherwise, will return a list of elements that match the
    criteria.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        tags (str or list of str or set): Tag(s) to search for in this ElementTree.
        attribs (None or dict of str): Element attribute(s) to check against for a filtered element. A match is
            considered found only if all attributes match. Each attribute key should have a corresponding value with
            which to compare against.
        return_first (bool): Whether to immediately return once the first matching element is found.

    Returns:
        None or ET.Element or list of ET.Element: Matching element(s) found. Returns None if there was no match.
    """
    # Initialize return value
    elements = None if return_first else []

    # Make sure tags is list
    tags = [tags] if type(tags) is str else tags

    # Check the current element for matching conditions
    if root.tag in tags:
        matching = True
        if attribs is not None:
            for k, v in attribs.items():
                if root.get(k) != v:
                    matching = False
                    break
        # If all criteria were matched, add this to the solution (or return immediately if specified)
        if matching:
            if return_first:
                return root
            else:
                elements.append(root)
    # Continue recursively searching through the element tree
    for r in root:
        if return_first:
            elements = find_elements(tags=tags, attribs=attribs, root=r, return_first=return_first)
            if elements is not None:
                return elements
        else:
            found_elements = find_elements(tags=tags, attribs=attribs, root=r, return_first=return_first)
            pre_elements = deepcopy(elements)
            if found_elements:
                elements += found_elements if type(found_elements) is list else [found_elements]

    return elements if elements else None

