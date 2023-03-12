import io

from dataclasses import dataclass
from typing import Optional
from threading import Thread

import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

from ipynb.fs.defs.Trainer import Model, LabelTransformer, NORMALIZE_MEAN, NORMALIZE_DEVIATIONS, IMAGE_RESIZE
from ClipHandlerS import Clipboard, ClipTypes

transform = transforms.Compose([
    transforms.Resize(IMAGE_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_DEVIATIONS),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = True

class ClipProcessor():
    
    ALL_CLIP_TYPES = [ClipTypes.IMAGE, ClipTypes.TEXT, ClipTypes.RGBIMAGE]

    @dataclass
    class _ClipPrinterStruct:
        name : str
        rowStartDelimeter : Optional[str] = ''
        startMessage : Optional[str] = ''
        endMessage : Optional[str] = ''
        rowEndDelimeter : Optional[str] = ''
    
    __ClipTypesDict = {ClipTypes.IMAGE: _ClipPrinterStruct(
                                                    name="image",
                                                    startMessage="Accepted image from clipboard",
                                                    endMessage="Image processed",
                                                    rowStartDelimeter='_',
                                                    rowEndDelimeter='‾'),
                    ClipTypes.TEXT: _ClipPrinterStruct(
                                                    name="text",
                                                    rowStartDelimeter="=",
                    ),
                    ClipTypes.RGBIMAGE: _ClipPrinterStruct(
                                                    name="rgbimage",
                                                    startMessage="Accepted image from clipboard",
                                                    endMessage="Image processed",
                                                    rowStartDelimeter='_',
                    )}

    def clipPrinter(type: ClipTypes):
        def clipPrinterF(func):
            def _wrapper(*args, **kwargs):
                try:
                    clipDecor = ClipProcessor.__ClipTypesDict[type]
                    if clipDecor.rowStartDelimeter: print(clipDecor.rowStartDelimeter*50)
                    print(f"\t{clipDecor.startMessage}\n") if clipDecor.startMessage else print()
                    func(*args, **kwargs)
                    print(f"\n\t{clipDecor.endMessage}") if clipDecor.endMessage else print()
                except Exception as e:
                    print(f"[!!] Error ocuired: {e}")
                finally:
                    if clipDecor.rowEndDelimeter: print(clipDecor.rowEndDelimeter*50)
                    elif clipDecor.rowStartDelimeter: print(clipDecor.rowStartDelimeter*50)
                    print()
            return _wrapper
        return clipPrinterF

    def __init__(self):
        self.CLIPS_PROCESSED = 0
        self.TEXT_PROCESSED = 0
        self.IMAGE_PROCESSED = 0
        self._processingTypes = ClipProcessor.ALL_CLIP_TYPES
        self.lbTrans = LabelTransformer()
        self.model = Model()
        self.model.load_state_dict(torch.load("01_Task\w.pth"))
        self.model.eval()
        self.softMax = nn.Softmax(dim=0)

    def setTypes(self, types : list[ClipTypes]):
        self._processingTypes = types

    def __call__(self, clip, *args, **kwds):
        return self.processClip(clip)

    def processClip(self, clip):
        if not clip.type in self._processingTypes: 
            return
        self.CLIPS_PROCESSED += 1
        if clip.type == ClipTypes.IMAGE:
            return self.processImage(clip.value)
        elif clip.type == ClipTypes.TEXT:
            return self.processText(clip.value)
        elif clip.type == ClipTypes.RGBIMAGE:
            return self.processImage(clip.value)
        else:
            return None

    @clipPrinter(ClipTypes.IMAGE)
    def processImage(self, clip):
        self.IMAGE_PROCESSED += 1
        img = Image.open(io.BytesIO(clip))
        img_jpg = img.convert('RGB')
        image = transform(img_jpg)
        image = image.reshape(1, *image.shape)
        image.to(device)
        with torch.no_grad():
            pred = self.model(image)
            pred_ind =  pred.max(dim = 1, keepdim=True)[1]
            print(f"{self.lbTrans(int(pred_ind[0][0]))} with {float(self.softMax(pred[0])[pred_ind])*100:.2f}%")

    @clipPrinter(ClipTypes.TEXT)
    def processText(self, clip):
        self.TEXT_PROCESSED += 1
        print(clip)

    
if __name__ == '__main__':
    clpProc = ClipProcessor()    

    if DEBUG: print(f"Starting clipboard daemon..")
    clipboard = Clipboard(on_update=clpProc, check_start_clip=True)
    clipThread = Thread(target=clipboard.listen, daemon=True)
    if DEBUG: print(f"Clipboard daemon state: {clipThread}")
    clipThread.start()
    if DEBUG: print(f"Clipboard daemon state set to 'work'")
    while True: pass