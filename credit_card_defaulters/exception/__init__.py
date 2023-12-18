import os
import sys

class CreditException(Exception):  ## inheritance of parent class Exception

    def __init__(self, error_message:Exception, error_detail:sys):   ## creating object of exception as error messgage and sys module as error detail
        super().__init__(error_message)   ### or Exception(error_message)     
        ## super() means to parent class we are passing informaion error message which we got inside child class
        """
        
        Using super().__init__(error_message) in the context of an __init__ method typically means 
        that you are calling the constructor of the parent class (superclass) and passing the error_message 
        as an argument to it. The usage of super() is often seen in class inheritance scenarios where a child 
        class wants to invoke the constructor of its parent class. Here's a breakdown of the difference:
        
        passing (error_message) as argeument inside super(). is done to initiialize parent class with certian
        paramenter that the constructer needs in parent class

        super().__init__(parent_arg) is used in the ChildClass to call the constructor of ParentClass and 
        pass the required parent_arg. This ensures that both the parent and child classes are properly 
        initialized.
        
        When you use super().__init__(...), it only initializes the constructor of the immediate parent class. 
        It does not initialize the constructors of all ancestor classes.

        """
        
        
        self.error_message = error_message           

    @staticmethod          ## no object of class is needed , we can call funtion belw using class name.funct name
    def get_detailed_error_message(error_message:Exception, error_detail:sys)-> str:   ## this function is going to return a string

        """
        error message : Exception object
        error detail : object of sys module 
             
        """
        _,_, exec_tb = error_detail.exc_info()  ## exec_tb = exception traceback
                  ## exc_info return excption info about error details in 3 tuples params-> (type,value,traceback)
                  ## we are skipping the frst two info typ and value, that s why we left blank but we want trace back in varivble exec_tb
                  ## from tackeback we extract line number and file name

        """
        In exception handling in Python, the sys module is not directly used for handling exceptions itself, 
        but it can be indirectly involved in providing information about the exception. The sys module provides 
        access to some variables used or maintained by the Python interpreter, and it can be used to access 
        information about the current exception being handled.
        One common use of the sys module in exception handling is to access information about the exception 
        using sys.exc_info(). This function returns a tuple containing information about the current exception 
        being handled. The tuple includes the exception type, the exception value, and the traceback object.
        
        """


        file_name = exec_tb.tb_frame.f_code.co_filename
        line_number = exec_tb.tb_frame.f_lineno

        error_message = f"Error occured in script :[{file_name}] at line number [{line_number}] error message :[{error_message}]"
        return error_message


    def __str__(self): ## this str() returns the string when we give print statment for the class() 
        return self.error_message
    
    """

     If you want to create a string representation for a custom object, you can define the __str__ method in 
     your class. The str() function will then call this method when converting an instance of your class to 
     a string

    """

    def __repr__(self) -> str: ## this repr() will return something if obhject of calss is called
        return CreditException.__name__.__str()





