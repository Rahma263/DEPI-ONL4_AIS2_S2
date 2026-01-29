class Course:
    _id_counter = 1
    def __init__(self,name):
        self.course_id = Course._id_counter
        Course._id_counter += 1
        self.name = name
        self.enrolled_student = []
    def __repr__(self):
        return f"course id: {self.course_id} , Name : {self.name} , Enrolled :{len(self.enrolled_student)}"
    def enroll_student(self, student):
        if student not in self.enrolled_student:
            self.enrolled_student.append(student)
            print("the student enrolled successfully")
        else:
            print("student already enrolled")
    def remove_student(self,student):
        if student in self.enrolled_student:
            self.enrolled_student.remove(student)
            print("the student is removed successfully")
        else:
            print("the student not found in this course")
        