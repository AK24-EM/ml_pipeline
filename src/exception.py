import sys


def error_detail_exception(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occurred in file [{0}] at line [{1}]: {2}".format(
        filename, line_number, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_detail_exception(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
