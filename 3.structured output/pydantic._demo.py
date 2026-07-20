from pydantic import BaseModel,EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str ='Yashvi'
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0 , lt=10)

new_student = {'age' : 20, 'email' : 'yashvishrest@gmail.com', 'cgpa':9.0}

student = Student(**new_student)

print(student)