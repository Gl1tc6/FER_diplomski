from Menu import Menu

menu = Menu()

while True:
    mode = menu.MainMenu()
    if mode == "Data Source":
        menu.FilePathInput()
    elif mode == "Topics":
        menu.TopicSelection()
    elif mode == "Mode":
        menu.ModeSelection()
    elif mode == "Start":
        menu.GetData()
        menu.Quizlet()