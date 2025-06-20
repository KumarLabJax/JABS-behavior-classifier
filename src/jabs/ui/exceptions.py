class ThreadTerminatedError(RuntimeError):
    """Exception raised when thread is cancelled by the user."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.message = "Thread was cancelled by the user" if not args else args[0]
