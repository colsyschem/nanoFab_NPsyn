import settings
import serial
from rich import print
import time

class Ismatec:
    def __init__(self):
        self.ser = serial.Serial(
            port=settings.PORT,
            baudrate=settings.BAUDRATE,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=1
        )
        self.channels = {
            1: {
                'volume': self.get_volume(1),
                'time': self.get_time(1),
                'rotation': self.get_rotation(1),
                'delay': settings.DELAY[0]
            },
            2: {
                'volume': self.get_volume(2),
                'time': self.get_time(2),
                'rotation': self.get_rotation(2),
                'delay': settings.DELAY[1]
            },
            3: {
                'volume': self.get_volume(3),
                'time': self.get_time(3),
                'rotation': self.get_rotation(3),
                'delay': settings.DELAY[2]
            },
            4: {
                'volume': self.get_volume(4),
                'time': self.get_time(4),
                'rotation': self.get_rotation(4),
                'delay': settings.DELAY[3]
            },
        }

    def get_volume(self, channel: int) -> str:
        response = self.send_command(f"{channel}v\r\n")

        return f"{float(response):.2f}"
    
    def get_time(self, channel: int) -> float:
        response = self.send_command(f"{channel}xT\r\n")

        return f"{float(response)/10:.2f}"
    
    def get_rotation(self, channel: int) -> int:
        response = self.send_command(f"{channel}xD\r\n")
        if response == "J\r\n":
            return 0
        else:
            return 1
    
    def set_volume(self, volume: float, channel: int):
        self.channels[channel]['volume'] = volume

        s = f"{abs(volume/(10**3)):.3e}"

        self.send_command(f"{channel}v{s[0]}{s[2:5]}{s[-3]}{s[-1]}\r\n")

        print(f"\n[bold green][OK] Volume ({channel}) modified to: {volume}.[/bold green]\n")
    
    def set_time(self, time: float, channel: int):
        self.channels[channel]['time'] = time

        self.send_command(f"{channel}xT{time*10:08}\r\n")

        print(f"\n[bold green][OK] Time ({channel}) modified to: {time}.[/bold green]\n")

    def set_rotation(self, rotation: int, channel: int):
        self.channels[channel]['rotation'] = rotation

        if rotation == 0:
            x = "J"
            trotation = "Clockwise"
        else:
            x = "K"
            trotation = "Counter-Clockwise"

        self.send_command(f"{channel}{x}\r\n")

        print(f"\n[bold green][OK] Rotation ({channel}) modified to: {trotation}.[/bold green]\n")

    def send_command(self, command: str) -> str:
        self.ser.write(command.encode())
        response = self.ser.readline().decode()
        
        return response
    
    def run(self):
        for i in self.channels.keys():
            self.send_command(f"{i}H\r\n")

            time.sleep(int(float(self.get_time(i))) + settings.DELAY[i-1])
    
    def close_serial_connection(self):
        
        return self.ser.close()
    