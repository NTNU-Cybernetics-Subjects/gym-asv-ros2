import time


class Timer:

    def __init__(self) -> None:
        self.start_time = time.time()

    def tic(self):
        self.start_time = time.time()

    def toc(self) -> float:
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        return elapsed_time
    

    @staticmethod
    def prettify(elapsed_time: float) -> str:
        """Makes the time into a nice string on the format Hour:Minuts:seconds"""
        elapsed_time_struct = time.gmtime(elapsed_time)
        formatted_time = time.strftime("%H:%M:%S", elapsed_time_struct)
        return formatted_time

if __name__ == "__main__":

    t = Timer()
    time.sleep(1.2)
    el = t.toc()
    print(t.prettify(el))
    print(el)
