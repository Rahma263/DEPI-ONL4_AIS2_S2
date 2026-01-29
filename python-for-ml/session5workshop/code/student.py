"""
this is a class for a student
"""
class Student:
    _id_counter = 1  
    
    def __init__(self,name):
        self.student_id = Student._id_counter #because the variable is out the scope of the constructor so we put Student._id_counter
        Student._id_counter += 1   #increment by 1 when i make a object 
        self.name = name  
        self.grades = {}
        self.enrolled_courses = []
    def __str__(self):
        return f"Student ID: {self.student_id} , Name {self.name} , Grade: {self.grades}"
    def __repr__(self):
        return f"Student ID: {self.student_id} , Name {self.name} , Grade: {self.grades}"
    def add_grade(self,course_id ,grade):
        self.grades[course_id] = grade
    def enroll_in_course(self,course):
        self.enrolled_courses.append(course)
    
    