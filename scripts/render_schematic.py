#!/usr/bin/env python3
import sys
from pathlib import Path
import ctypes
import ctypes.util
import gi

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
handle = Rsvg.Handle.new_from_file(str(source))
success, width, height = handle.get_intrinsic_size_in_pixels()
if not success or width <= 0 or height <= 0:
    raise ValueError("SVG requires explicit positive intrinsic dimensions")
cairo_library = ctypes.CDLL(ctypes.util.find_library("cairo"))
rsvg_library = ctypes.CDLL(ctypes.util.find_library("rsvg-2"))
object_library = ctypes.CDLL(ctypes.util.find_library("gobject-2.0"))
glib_library = ctypes.CDLL(ctypes.util.find_library("glib-2.0"))


class Rectangle(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in ("x", "y", "width", "height")]


class Error(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("code", ctypes.c_int), ("message", ctypes.c_char_p)]


def bind(library, name, result, arguments):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = arguments
    return function


pointer = ctypes.c_void_p
error_pointer = ctypes.POINTER(Error)
new_handle = bind(rsvg_library, "rsvg_handle_new_from_file", pointer,
                  [ctypes.c_char_p, ctypes.POINTER(error_pointer)])
render = bind(rsvg_library, "rsvg_handle_render_document", ctypes.c_int,
              [pointer, pointer, ctypes.POINTER(Rectangle), ctypes.POINTER(error_pointer)])
new_surface = bind(cairo_library, "cairo_pdf_surface_create", pointer,
                   [ctypes.c_char_p, ctypes.c_double, ctypes.c_double])
new_context = bind(cairo_library, "cairo_create", pointer, [pointer])
finish = bind(cairo_library, "cairo_surface_finish", None, [pointer])
surface_status = bind(cairo_library, "cairo_surface_status", ctypes.c_int, [pointer])
destroy_context = bind(cairo_library, "cairo_destroy", None, [pointer])
destroy_surface = bind(cairo_library, "cairo_surface_destroy", None, [pointer])
unref = bind(object_library, "g_object_unref", None, [pointer])
free_error = bind(glib_library, "g_error_free", None, [error_pointer])
error = error_pointer()
native_handle = new_handle(str(source).encode(), ctypes.byref(error))
if not native_handle:
    message = error.contents.message.decode() if error else "Cannot load SVG"
    if error:
        free_error(error)
    raise RuntimeError(message)
surface = new_surface(str(destination).encode(), width * 0.75, height * 0.75)
context = new_context(surface)
try:
    viewport = Rectangle(0, 0, width * 0.75, height * 0.75)
    if not render(native_handle, context, ctypes.byref(viewport), ctypes.byref(error)):
        message = error.contents.message.decode() if error else "Rsvg vector rendering failed"
        if error:
            free_error(error)
        raise RuntimeError(message)
    finish(surface)
    if surface_status(surface):
        raise RuntimeError("Cairo PDF surface reported an output error")
finally:
    destroy_context(context)
    destroy_surface(surface)
    unref(native_handle)
print(f"Rendered vector PDF: {destination.name}")
