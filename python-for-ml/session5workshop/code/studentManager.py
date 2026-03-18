from course import Course
from student import Student

class SystemManager:
        
    def __init__(self):
        self.students = {}
        self.courses = {}
        
    def add_student(self,name):
        student = Student(name)
        self.students[student.student_id] = student #student.student_id the firststudent is object of Student
        print("the student added successfully")
        return student.student_id
    
    def remove_student(self,student_id):
        if student_id in self.students:
            student = self.students[student_id]
            if not student.enrolled_courses:
                del self.students[student_id]
                print("the student removed successfully")
            else:
                print("the student has enrolled course  . cannot removed")
        else:
            print("invaild id")
            
    def add_course(self,name):
        course = Course(name)
        self.courses[course.course_id] = course
        print("the course added successfully!")
        return course.course_id
    def remove_course(self,course_id):
        if course_id in self.courses:
           course = self.courses[course_id]
           if not course.enrolled_student:
               del self.courses[course_id]
               print("the course removed ")
           else:
               print("the course cannot removed")
        else:
            print("invaild course ID")
        
    
        