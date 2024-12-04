import random
import pygame
import tkinter
import tkinter.filedialog
import sys
import os
import glob
from pygame.rect import Rect

class Menu:
    def __init__(self) -> None:
        self.FilePath = ""
        self.Mode = ""
        self.scroll_y = 0
        self.Topics = []
        self.Data = []
        self.Answers = []

    def SelectAllTopics(self,topics):
        self.Topics = topics

    def SelectTopic(self,topic):
        self.Topics.append(topic)

    def RemoveTopic(self,topic):
        self.Topics.remove(topic)

    def PromptFile(self):
        top = tkinter.Tk()
        top.withdraw()
        self.FilePath = tkinter.filedialog.askdirectory(parent=top)
        top.destroy()
        return 

    def MouseDownEvent(self,buttons):
        for button in buttons:

            buttonData = button[0]
            buttonAction = button[1]
            buttonArgs = button[2]

            if buttonData.left <= self.mousePosition[0] <= buttonData.right and buttonData.top+self.scroll_y <= self.mousePosition[1] <= buttonData.bottom + self.scroll_y: 
                
                return buttonAction(buttonArgs)

    def QuitGameReturnPressedButton(self,buttonName = None):
        self.run = False
        if buttonName == None:
            return
        return buttonName
    
    def DrawButtonWithText(self,screen,x_pos,y_pos,width,height,primaryColor,hoverColor,text):

        if x_pos <= self.mousePosition[0] <= x_pos+width and y_pos+self.scroll_y <= self.mousePosition[1] <= y_pos+height+self.scroll_y: 
            rect = pygame.draw.rect(screen,hoverColor,[x_pos,y_pos,width,height]) 
        else: 
            rect = pygame.draw.rect(screen,primaryColor,[x_pos,y_pos,width,height])
        text = pygame.font.SysFont('Corbel',20).render(text , True , (0,0,0))
        screen.blit(text , text.get_rect(center = rect.center))

        return rect

    def CenterInfoText(self,screen,y_pos,textSize,textData):
        screenWidth = screen.get_rect().width
        chunkSize = screenWidth//(textSize//2)
        if len(textData) > chunkSize:
            iteration = 0
            while True:
                startIndex = chunkSize*iteration
                endIndex = chunkSize*(iteration+1)
                if endIndex >= len(textData):
                    endIndex = len(textData)-1
                    textChunk = textData[startIndex:endIndex]
                    text = pygame.font.SysFont('Corbel',textSize).render(textChunk , True , (255,255,255))
                    screen.blit(text , text.get_rect(centerx = screen.get_rect().centerx,centery = y_pos+iteration*textSize))
                    break
                textChunk = textData[startIndex:endIndex]
                text = pygame.font.SysFont('Corbel',textSize).render(textChunk , True , (255,255,255))
                screen.blit(text , text.get_rect(centerx = screen.get_rect().centerx,centery = y_pos+iteration*textSize))
                iteration += 1
        else:
            text = pygame.font.SysFont('Corbel',textSize).render(textData , True , (255,255,255))
            screen.blit(text , text.get_rect(centerx = screen.get_rect().centerx,centery = y_pos))

    
        

    def MainMenu(self):

        pygame.init()
        display = (200, 200)
        screen = pygame.display.set_mode(display)
        self.run = True
        buttons = []
        while self.run:

            for ev in pygame.event.get(): 
          
                if ev.type == pygame.QUIT:
                    
                    self.run = False
                    pygame.quit()
                    sys.exit()
                    
                if ev.type == pygame.MOUSEBUTTONDOWN:  

                    returnData = self.MouseDownEvent(buttons)
                        
            buttons.clear()
                        
            screen.fill((0,0,0)) 
            self.mousePosition = pygame.mouse.get_pos()

            #Create data source button
            if self.FilePath == "":
                buttons.append([self.DrawButtonWithText(screen,0,0,200,40,(255,255,255),(200,200,200),"Choose Data Source"),lambda x: self.QuitGameReturnPressedButton("Data Source"),None])
            else:
                buttons.append([self.DrawButtonWithText(screen,0,0,200,40,(111,194,118),(200,200,200),"Choose Data Source"),lambda x: self.QuitGameReturnPressedButton("Data Source"),None])

            #Create topics button
            if self.FilePath == "":
                buttons.append([self.DrawButtonWithText(screen,0,40,200,40,(255,155,155),(200,200,200),"Choose Topics"),lambda x: x,None])
            elif len(self.Topics) == 0:
                buttons.append([self.DrawButtonWithText(screen,0,40,200,40,(255,255,255),(200,200,200),"Choose Topics"),lambda x: self.QuitGameReturnPressedButton("Topics"),None])
            else:
                buttons.append([self.DrawButtonWithText(screen,0,40,200,40,(111,194,118),(200,200,200),"Choose Topics"),lambda x: self.QuitGameReturnPressedButton("Topics"),None])

            #Create mode button
            if self.FilePath == "":
                buttons.append([self.DrawButtonWithText(screen,0,80,200,40,(255,155,155),(200,200,200),"Choose Mode"),lambda x: x,None])
            elif self.Mode == "":
                buttons.append([self.DrawButtonWithText(screen,0,80,200,40,(255,255,255),(200,200,200),"Choose Mode"),lambda x: self.QuitGameReturnPressedButton("Mode"),None])
            else:
                buttons.append([self.DrawButtonWithText(screen,0,80,200,40,(111,194,118),(200,200,200),"Choose Mode"),lambda x: self.QuitGameReturnPressedButton("Mode"),None])
            

            #Create start button
            if self.Mode == "" or len(self.Topics) == 0:
                buttons.append([self.DrawButtonWithText(screen,0,120,200,40,(255,155,155),(200,200,200),"Start"),lambda x: x,None])
            else:
                buttons.append([self.DrawButtonWithText(screen,0,120,200,40,(255,255,255),(200,200,200),"Start"),lambda x: self.QuitGameReturnPressedButton("Start"),None])
            
            pygame.display.update()
        
        pygame.quit()
        return returnData

    def FilePathInput(self):
        pygame.init()
        display = (300, 210)
        screen = pygame.display.set_mode(display)
        self.run = True
        buttons = []
        self.FilePath = ""
        while self.run:
            for ev in pygame.event.get(): 
          
                if ev.type == pygame.QUIT:
                    self.run = False
                    pygame.quit()
                    sys.exit()
                
                if ev.type == pygame.MOUSEBUTTONDOWN: 
                    
                    returnData = self.MouseDownEvent(buttons)
                
                if ev.type == pygame.KEYDOWN: 
                    if ev.key == pygame.K_RETURN:
                        self.run = False
            
            buttons.clear()
            screen.fill((0,0,0)) 
            self.mousePosition = pygame.mouse.get_pos()

            #screen.blit(pygame.font.SysFont('Corbel',20).render('Please select data source :' , True , (255,255,255)) , (screen.get_rect().centerx,10))
            self.CenterInfoText(screen,30,20,"Please select data source :")
            self.CenterInfoText(screen,50,20,self.FilePath)

            #Create select file button
            buttons.append([self.DrawButtonWithText(screen,50,130,200,40,(255,255,255),(200,200,200),"Select Directory"),lambda x: self.PromptFile(),None])

            #Create done button
            buttons.append([self.DrawButtonWithText(screen,50,170,200,40,(255,255,255),(200,200,200),"Done"),lambda x: self.QuitGameReturnPressedButton(),None])

            pygame.display.flip()
        
        pygame.quit()
        return
    
    def TopicSelection(self):
        allTopics = [x[0] for x in os.walk(self.FilePath)][1:]
        pygame.init()
        display = (500, 500)
        intermediate = pygame.surface.Surface((500, (len(allTopics)+2)*40))
        screen = pygame.display.set_mode(display)
        self.run = True
        buttons = []
        self.scroll_y = 0
        while self.run:

            for ev in pygame.event.get(): 
          
                if ev.type == pygame.QUIT:
                    
                    self.run = False
                    pygame.quit()
                    sys.exit()
                    
                if ev.type == pygame.MOUSEBUTTONDOWN:  
                    if ev.button == 1:
                        returnData = self.MouseDownEvent(buttons)
                    if ev.button == 4:
                        self.scroll_y = min(self.scroll_y + 15, 0)
                    if ev.button == 5:
                        self.scroll_y = max(self.scroll_y - 15, -300)

            buttons.clear()
                        
            screen.fill((0,0,0)) 
            self.mousePosition = pygame.mouse.get_pos()

            buttons.append([self.DrawButtonWithText(intermediate,0,0,500,40,(255,255,255),(200,200,200),"Done"),lambda x: self.QuitGameReturnPressedButton(),None])
              
            buttons.append([self.DrawButtonWithText(intermediate,0,40,500,40,(255,255,255),(200,200,200),"SelectAll"),lambda x: self.SelectAllTopics(allTopics.copy()),None])
                
            buttonHeight = 80
            for topic in allTopics:
                if topic in self.Topics:
                    buttons.append([self.DrawButtonWithText(intermediate,0,buttonHeight,500,40,(111,194,118),(200,200,200),os.path.split(topic)[-1]),lambda x: self.RemoveTopic(x),topic])
                else:
                    buttons.append([self.DrawButtonWithText(intermediate,0,buttonHeight,500,40,(255,255,255),(200,200,200),os.path.split(topic)[-1]),lambda x: self.SelectTopic(x),topic])
                buttonHeight+=40

            screen.blit(intermediate, (0, self.scroll_y))
            pygame.display.update()
        
        pygame.quit()

    def ModeSelection(self):
        pygame.init()
        display = (300, 300)
        screen = pygame.display.set_mode(display)
        self.run = True
        buttons = []
        self.scroll_y = 0
        while self.run:

            for ev in pygame.event.get(): 
          
                if ev.type == pygame.QUIT:
                    
                    self.run = False
                    pygame.quit()
                    sys.exit()
                    
                if ev.type == pygame.MOUSEBUTTONDOWN:  
                    if ev.button == 1:
                        self.Mode = self.MouseDownEvent(buttons)

            buttons.clear()
                        
            screen.fill((0,0,0)) 
            self.mousePosition = pygame.mouse.get_pos()

            buttons.append([self.DrawButtonWithText(screen,0,0,300,40,(255,255,255),(200,200,200),"All Ordered"),lambda x: self.QuitGameReturnPressedButton("All Ordered"),None])
              
            buttons.append([self.DrawButtonWithText(screen,0,40,300,40,(255,255,255),(200,200,200),"All Random"),lambda x: self.QuitGameReturnPressedButton("All Random"),None])
            
            buttons.append([self.DrawButtonWithText(screen,0,80,300,40,(255,255,255),(200,200,200),"Test Like"),lambda x: self.QuitGameReturnPressedButton("Test Like"),None])
                
            pygame.display.update()
        
        pygame.quit()

    def GetData(self):
        self.counter = 0
        for topicPath in self.Topics:
            
            topicData = [os.path.abspath(x) for x in glob.glob(topicPath+"\\**\\*.png", recursive=True)]
            answersFile = open(topicPath + "\\Answers.txt",'r')
            answerData = [x.strip() for x in answersFile.readlines()]
            self.Data.append(topicData)
            self.Answers.append(answerData)

    def Quizlet(self):
        if self.Mode == "All Ordered":
            for topicIndex,topic in enumerate(self.Data):
                for questionIndex,question in enumerate(topic):
                    
                    self.counter += 1
                    self.correctAnswer = self.Answers[topicIndex][questionIndex]
                    self.selectedAnswer = None
                    self.AskQuestion(question)

        if self.Mode == "All Random":
            while any(self.Data):
                self.counter += 1
                topicIndex = random.randint(0, len(self.Data)-1)
                topic = self.Data[topicIndex]
                while not any(topic):
                    topicIndex = random.randint(0, len(self.Data)-1)
                    topic = self.Data[topicIndex]
                questionIndex = random.randint(0, len(topic)-1)
                question = topic[questionIndex]
                self.correctAnswer = self.Answers[topicIndex][questionIndex]
                self.selectedAnswer = None
                self.AskQuestion(question)
                self.Data[topicIndex].pop(questionIndex)
                self.Answers[topicIndex].pop(questionIndex)

        if self.Mode == "Test Like":
            for topicIndex,topic in enumerate(self.Data):
                numberOfQuesitons = random.randint(1, 2)
                for i in range(0,numberOfQuesitons):
                    self.counter += 1
                    questionIndex = random.randint(0, len(topic)-1)
                    question = topic[questionIndex]
                    self.correctAnswer = self.Answers[topicIndex][questionIndex]
                    self.selectedAnswer = None
                    self.AskQuestion(question)
                    self.Data[topicIndex].remove(question)


    def AskQuestion(self,question):
        pygame.init()
        display = (1100, 700)
        screen = pygame.display.set_mode(display)
        self.run = True
        buttons = []
        while self.run:

            for ev in pygame.event.get(): 
          
                if ev.type == pygame.QUIT:
                    
                    self.run = False
                    pygame.quit()
                    sys.exit()
                    
                if ev.type == pygame.MOUSEBUTTONDOWN:  
                    if ev.button == 1:
                        returnData = self.MouseDownEvent(buttons)

            buttons.clear()
                        
            screen.fill((0,0,0)) 
            self.mousePosition = pygame.mouse.get_pos()

            buttons.append([self.DrawButtonWithText(screen,0,0,20,20,(255,255,255),(100,136,234),str(self.counter)),lambda x: x, None])
            
            
            picture = pygame.image.load(question)
            picture = pygame.transform.scale(picture, (900, 500))
            rect = picture.get_rect(centerx = screen.get_rect().centerx, centery = 300)
            screen.blit(picture, rect)

            if self.selectedAnswer == None or (self.selectedAnswer != "A" and self.correctAnswer != "A"):
                buttons.append([self.DrawButtonWithText(screen,10,600,200,100,(255,255,255),(100,136,234),"A"),lambda x: self.SelectAnswer("A"), None])
            elif self.selectedAnswer == "A" and self.correctAnswer != "A":
                buttons.append([self.DrawButtonWithText(screen,10,600,200,100,(255,155,155),(255,155,155),"A"),lambda x: self.SelectAnswer("A"), None])
            elif self.correctAnswer == "A":
                buttons.append([self.DrawButtonWithText(screen,10,600,200,100,(111,194,118),(111,194,118),"A"),lambda x: self.SelectAnswer("A"), None])
            

            
            if self.selectedAnswer == None or (self.selectedAnswer != "B" and self.correctAnswer != "B"):
                buttons.append([self.DrawButtonWithText(screen,230,600,200,100,(255,255,255),(100,136,234),"B"),lambda x: self.SelectAnswer("B"), None])
            elif self.selectedAnswer == "B" and self.correctAnswer != "B":
                buttons.append([self.DrawButtonWithText(screen,230,600,200,100,(255,155,155),(255,155,155),"B"),lambda x: self.SelectAnswer("B"), None])
            elif self.correctAnswer == "B":
                buttons.append([self.DrawButtonWithText(screen,230,600,200,100,(111,194,118),(111,194,118),"B"),lambda x: self.SelectAnswer("B"), None])
            
            if self.selectedAnswer == None or (self.selectedAnswer != "C" and self.correctAnswer != "C"):
                buttons.append([self.DrawButtonWithText(screen,450,600,200,100,(255,255,255),(100,136,234),"C"),lambda x: self.SelectAnswer("C"), None])
            elif self.selectedAnswer == "C" and self.correctAnswer != "C":
                buttons.append([self.DrawButtonWithText(screen,450,600,200,100,(255,155,155),(255,155,155),"C"),lambda x: self.SelectAnswer("C"), None])
            elif self.correctAnswer == "C":
                buttons.append([self.DrawButtonWithText(screen,450,600,200,100,(111,194,118),(111,194,118),"C"),lambda x: self.SelectAnswer("C"), None])
            
            if self.selectedAnswer == None or (self.selectedAnswer != "D" and self.correctAnswer != "D"):
                buttons.append([self.DrawButtonWithText(screen,670,600,200,100,(255,255,255),(100,136,234),"D"),lambda x: self.SelectAnswer("D"), None])
            elif self.selectedAnswer == "D" and self.correctAnswer != "D":
                buttons.append([self.DrawButtonWithText(screen,670,600,200,100,(255,155,155),(255,155,155),"D"),lambda x: self.SelectAnswer("D"), None])
            elif self.correctAnswer == "D":
                buttons.append([self.DrawButtonWithText(screen,670,600,200,100,(111,194,118),(111,194,118),"D"),lambda x: self.SelectAnswer("D"), None])
            
            
            buttons.append([self.DrawButtonWithText(screen,890,600,200,100,(255,255,255),(100,136,234),"Next"),lambda x: self.NextQuestion(), None])

            pygame.display.update()
        
        pygame.quit()

    def SelectAnswer(self,answer):
        self.selectedAnswer = answer
    
    def NextQuestion(self):
        self.run = False

    
