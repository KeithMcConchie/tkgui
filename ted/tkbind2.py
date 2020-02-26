import tkinter as tk
def on_control_o(event=None):
    text.insert(1.0, 'aaa')
    # return "break"
win = tk.Tk()
text = tk.Text(win, fg="black", bg="white")
text.bind("<Control-o>", on_control_o)
text.pack()


# text.bind("<Control-o>", lambda e, t=text: t.insert(1.0, 'aaa'))


win.mainloop()
