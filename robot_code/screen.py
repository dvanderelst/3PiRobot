from pololu_3pi_2040_robot import robot

class Screen:
    def __init__(self):
        self.display = robot.Display()
        self.y_positions = [0, 13, 23, 33, 47, 57]
        self.texts = [''] * 6
    
    def write(self, line_nr, text):
        if line_nr < 0: line_nr = 0
        if line_nr > 5: line_nr = 5
        text = str(text)
        self.texts[line_nr] = text
        self.update()
            
    def clear(self):
        self.texts = [''] * 6
        self.update()
    
    def update(self):
        self.display.fill(0)
        for i in range(6):
            y_position = self.y_positions[i]
            text = self.texts[i]
            self.display.text(text, 0, y_position)
        self.display.show()
        
        
        
        
        
        