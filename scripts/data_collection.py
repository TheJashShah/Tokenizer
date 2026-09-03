'''
Provides a class to collect text data for Languages (Normal and Programming).

desired output:

obj1 = DataCollector()
objj1.add_language("english", size=100) # collects 100MB of English language data.
obj1.add_programming_language("python", size=100) # collects 100MB of Python code.

Provides the text in either a single text file, or a folder with a seperate file for each language and programming language.
'''

class DataCollector:
    def __init__(self, output_dir):
        
        self.languages = {}
        self.programming_languages = {}
        
    def add_language(self, language, size):
        
        '''
        Adds `size` MB of language.
        '''
        
    def add_programming_language(self, language, size):
        
        '''
        Adds `size` MB of programming language.
        '''
        
    def save_to_disk(self, output_dir=None, output_file=None):
        
        '''
        Saves all data to disk, either in the form of a single file, or in a folder, as seperate files.
        '''
        
        
        
        
         
        