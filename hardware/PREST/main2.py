from pyfiglet import Figlet
from ismatec2 import Ismatec
from rich import print
import watcher
import settings
import inquirer

def main(obj):
    custom_fig = Figlet(font='standard')
    print(custom_fig.renderText('Reglo-ICC'))

    start_menu = [
        inquirer.List('mode',
                      message="Select mode",
                      choices=['Automatic', 'Manual', 'Exit'],
                      ),
    ]
    start_menu_selected = inquirer.prompt(start_menu)

    main_menu = [
        inquirer.List('action',
                      message="Select action",
                      choices=['Set Channel', 'Set Volume', 'Set Time', 'Set Rotation', 'Run', 'Back to Previous Menu', 'Exit'],
                      ),
    ]

    channel = 1
    while True:
        if start_menu_selected['mode'] == 'Automatic':
            print(f"Reading data from {settings.SOURCE_DIR}\n")
            
            event_handler = watcher.MyHandler(obj)
            observer = watcher.Observer()
            observer.schedule(event_handler, path=settings.SOURCE_DIR, recursive=False)
            observer.start()

            try:
                while True:
                    watcher.time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()

            start_menu_selected = inquirer.prompt(start_menu)
        elif start_menu_selected['mode'] == 'Manual':
            for i in range(1,5):
                print("+-------------------------------------------------------------------------------------------------+")
                print("| Channel =", '{:^5d}'.format(i), "Volume \[ul] =", '{:^9s}'.format(obj.get_volume(i)), "Time \[s] =", '{:^9s}'.format(obj.get_time(i)), "Rotation =",'{:^5d}'.format(obj.get_rotation(i)), "Delay \[s] =",'{:^5d}'.format(settings.DELAY[i-1]), "|")
            print("+-------------------------------------------------------------------------------------------------+")
            print(f"\n+--------------------+\n| Current Channel: {channel} |\n+--------------------+\n")

            main_menu_selected = inquirer.prompt(main_menu)
            match main_menu_selected['action']:
                case 'Set Channel':
                    tchannel = eval(input("Set Channel [1-4]: "))
                    if tchannel < 1 or tchannel > 4:
                        print("\n[bold yellow][WARNING] Sorry, channel must be between [1-4], try again.[/bold yellow]\n")
                    else:
                        channel = tchannel
                        print(f"\n[bold green][OK] Channel modified to: {channel}.[/bold green] \n")
                case 'Set Volume':
                    try:
                        volume = eval(input("Set Volume [ul]: "))
                        obj.set_volume(volume, channel)
                    except:
                        print("\n[bold yellow][WARNING] Sorry, volume must be numeric, try again.[/bold yellow]\n")
                case 'Set Time':
                    try:
                        time = eval(input("Set Time [s]: "))
                        obj.set_time(time, channel)
                    except Exception as e:
                        print(e)
                case 'Set Rotation':
                    try:
                        rotation = eval(input("Clockwise [0] / Counter-Clockwise [1]: "))
                        obj.set_rotation(rotation, channel)
                    except:
                        print("\n[bold yellow][WARNING] Sorry, rotation must be 0 or 1, try again.[/bold yellow]\n")
                case 'Run':
                    print("\n[bold]Starting pump with monitoring...[/bold]")
                    obj.run()
                    print("[bold green]All channels completed![/bold green]\n")
                case 'Back to Previous Menu':
                    start_menu_selected = inquirer.prompt(start_menu)
                case 'Exit':
                    obj.close_serial_connection()
                    break
        elif start_menu_selected['mode'] == 'Exit':
            obj.close_serial_connection()
            break

if __name__ == "__main__":
    reglo_icc = Ismatec()
    main(reglo_icc)