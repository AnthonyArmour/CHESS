import threading
import requests

class BackGroundWorker(object):
    """ Threading example class
    The run() method will be started and it will run in the background
    until the application exits.
    """

    def __init__(self, url):                                  # Start the execution
        self.mem = url+"/api/memory_buffer"
        self.save = url+"/api/save"
        self.load = url+"/api/load"


    def API_Save_Memory(self):
        thread = threading.Thread(target=self.Save_Buffer, args=())
        thread.daemon = True                           # Daemonize thread
        thread.start()

    def Save_Buffer(self):
        saved = requests.post(self.save, verify=False)
        print(saved)
