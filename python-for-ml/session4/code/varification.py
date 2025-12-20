import random

def varification_code(length:int):
    """
        this function take a length from the user and generate a random password 
        :param_1:the length of the pasword needed to be generated
        type param_1:int
        return:prin the password is generated 
    """
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*"
    password = ""
    for _ in range(length):
        password += random.choice(characters)
    print(f"genterate password: {password}")
