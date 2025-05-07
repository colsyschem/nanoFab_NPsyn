import time
import datetime
import settings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MyHandler(FileSystemEventHandler):
    def __init__(self, reglo_icc):
        super().__init__()
        self.reglo_icc = reglo_icc
        self.volumes = []
        self.last_run = datetime.datetime.now()

    def on_modified(self, event):
        if (datetime.datetime.now() - self.last_run).total_seconds() > 1:
            with open("./log/automatic.log", "a") as file:
                file.write(f'[+] New data detected in {event.src_path} at ({datetime.datetime.now()})\n')
            self.process_file(event.src_path)
            
            for channel, volume in enumerate(self.volumes):
                self.reglo_icc.set_volume(volume, channel + 1)
            
            self.reglo_icc.run()
            self.last_run = datetime.datetime.now()
    
    def process_file(self, path):
        try:
            with open(path, 'r') as file:
                lines = file.readlines()
                if lines:
                    line = lines[-1].strip()

                    self.volumes = [int(x) for x in line.split(',')[1:]]
                    print(f'[+] ({datetime.datetime.now()}) New data detected in {path}: {line}')
                    
        except Exception as e:
            print(f'Error reading file: {e}')

if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path=settings.SOURCE_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()