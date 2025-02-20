import sys
import traceback

class CustomException(Exception):
    def __init__(self, error, sys_obj):
        super().__init__(str(error))  # Convert error to string
        self.error_message = CustomException.get_error_message(error, sys_obj)

    @staticmethod
    def get_error_message(error, sys_obj):
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: {file_name}, line {line_number}: {str(error)}"

    def __str__(self):
        return self.error_message
