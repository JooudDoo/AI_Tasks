import ctypes
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union, List, Optional

import win32api, win32clipboard, win32con, win32gui
from ctypes.wintypes import *
from win32con import *

from datetime import datetime, timedelta

from enum import Enum

@dataclass
class BITMAPFILEHEADER(ctypes.Structure):
    _pack_ = 1  # structure field byte alignment
    _fields_ = [
        ('bfType', WORD),  # file type ("BM")
        ('bfSize', DWORD),  # file size in bytes
        ('bfReserved1', WORD),  # must be zero
        ('bfReserved2', WORD),  # must be zero
        ('bfOffBits', DWORD),  # byte offset to the pixel array
    ]
SIZEOF_BITMAPFILEHEADER = ctypes.sizeof(BITMAPFILEHEADER)

@dataclass
class BITMAPINFOHEADER(ctypes.Structure):
    _pack_ = 1  # structure field byte alignment
    _fields_ = [
        ('biSize', DWORD),
        ('biWidth', LONG),
        ('biHeight', LONG),
        ('biPLanes', WORD),
        ('biBitCount', WORD),
        ('biCompression', DWORD),
        ('biSizeImage', DWORD),
        ('biXPelsPerMeter', LONG),
        ('biYPelsPerMeter', LONG),
        ('biClrUsed', DWORD),
        ('biClrImportant', DWORD)
    ]
SIZEOF_BITMAPINFOHEADER = ctypes.sizeof(BITMAPINFOHEADER)

class ClipTypes(Enum):
    IMAGE = 1,
    RGBIMAGE = 10,
    TEXT = 2,
    FILES = 3

class Clipboard:
    _prevData = None

    @dataclass
    class Clip:
        type: ClipTypes
        value: Union[str, List[Path], bytes]
    
    def __init__(self,
            check_start_clip: bool = False,
            on_text: Callable[[str], None] = None,
            on_update: Callable[[Clip], None] = None, 
            on_files: Callable[[str], None] = None,
            on_image: Callable[[None], None] = None):
        self._check_start_clip = check_start_clip
        self._on_text = on_text
        self._on_update = on_update
        self._on_files = on_files
        self._on_image = on_image
    

    def _create_window(self) -> int:
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self._process_message
        wc.lpszClassName = self.__class__.__name__
        wc.hInstance = win32api.GetModuleHandle(None)
        class_atom = win32gui.RegisterClass(wc)
        return win32gui.CreateWindow(class_atom, self.__class__.__name__, 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)
    
    def _process_message(self, hwnd: int, msg: int, wparam: int, lparam: int):
        WM_CLIPBOARDUPDATE = 0x031D
        if msg == WM_CLIPBOARDUPDATE:
            self._process_clip()
        return 0
    
    def _process_clip(self):
        clip = self.read_clipboard()
        if not clip: return

        if self._prevData == clip.value:
            return
        self._prevData = clip.value

        if self._on_update:
            self._on_update(clip)
        elif clip.type == ClipTypes.TEXT and self._on_text:
            self._on_text(clip.value)
        elif clip.type == ClipTypes.FILES and self._on_text:
            self._on_files(clip.value)
        elif clip.type == ClipTypes.IMAGE and self._on_image:
            self._on_image(clip.value)
    
    @staticmethod
    def read_clipboard() -> Optional[Clip]:
        notOpenned = True
        try:
            win32clipboard.OpenClipboard()
            notOpenned = False
            
            def get_formatted(fmt):
                if win32clipboard.IsClipboardFormatAvailable(fmt):
                    return win32clipboard.GetClipboardData(fmt)
                return None

            if files := get_formatted(win32con.CF_HDROP):
                return Clipboard.Clip(ClipTypes.FILES, [Path(f) for f in files])
            elif text := get_formatted(win32con.CF_UNICODETEXT):
                return Clipboard.Clip(ClipTypes.TEXT, text)
            elif text_bytes := get_formatted(win32con.CF_TEXT):
                return Clipboard.Clip(ClipTypes.TEXT, text_bytes.decode())
            elif bitmap_handle := get_formatted(win32con.CF_DIB):
                bmih = BITMAPINFOHEADER()
                ctypes.memmove(ctypes.pointer(bmih), bitmap_handle, SIZEOF_BITMAPINFOHEADER)
                if bmih.biCompression == 0:
                    return Clipboard.Clip(ClipTypes.RGBIMAGE, bitmap_handle)
                if bmih.biCompression != BI_BITFIELDS:  # RGBA?
                    print('insupported compression type {}'.format(bmih.biCompression))
                    return None
                bmfh = BITMAPFILEHEADER()
                ctypes.memset(ctypes.pointer(bmfh), 0, SIZEOF_BITMAPFILEHEADER) 
                bmfh.bfType = ord('B') | (ord('M') << 8)
                bmfh.bfSize = SIZEOF_BITMAPFILEHEADER + len(bitmap_handle)
                SIZEOF_COLORTABLE = 0
                bmfh.bfOffBits = SIZEOF_BITMAPFILEHEADER + SIZEOF_BITMAPINFOHEADER + SIZEOF_COLORTABLE
                return Clipboard.Clip(ClipTypes.IMAGE, bytes(bmfh)+bitmap_handle)
            return None
        except Exception as e:
            print(f"Clipboard exception caused: {str(e)}")
        finally:
            if not notOpenned:
                win32clipboard.CloseClipboard()

    def listen(self):
        if self._check_start_clip:
            self._process_clip()
        
        def process():
            hwnd = self._create_window()
            ctypes.windll.user32.AddClipboardFormatListener(hwnd)
            win32gui.PumpMessages()
        
        th = threading.Thread(target=process, daemon=True)
        th.start()
        while th.is_alive():
            th.join(0.5)