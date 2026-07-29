from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """ 
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print("Name:", self.name)
        print("Age:", self.age)


# Create an object
student1 = Student("Alice", 20)

# Call the method
student1.display()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

chunk = splitter.split_text(text)
print(chunk)
print(len(chunk))