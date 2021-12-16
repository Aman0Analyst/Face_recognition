import logging
import sys

class DeepAttendanceLogger():
    def __init__(self,**kwargs):
        # logger = logging(**kwargs)
        return None
    
    @classmethod
    def get_logger(cls):
        log = logging.getLogger()

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(levelname)s] -> %(message)s",
            handlers=[
                logging.FileHandler("debug.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return log