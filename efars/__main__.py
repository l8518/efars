"EFARS Main - Starts the EFARS Prompt"
import app
import plugins

if __name__ == '__main__':
    PGM = app.EfarsPrompt(plugins)
    PGM.cmdloop()
    