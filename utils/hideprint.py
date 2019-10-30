import sys, os

class HiddenPrints:
    def __init__(self, bool=True):
        self.bool = bool
    def __enter__(self):
        if self.bool:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')#None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.bool:
            sys.stdout = self._original_stdout


if __name__ == '__main__':
    with HiddenPrints():
        print("This will not be printed")

    print("This will be printed as before")