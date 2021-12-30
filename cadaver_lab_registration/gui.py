import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

def loadFile():
    lbl_value_load["text"] = "Load file successfully"


def solve():
    lbl_value_solve["text"] = "Solving"


def output():
    lbl_value_output["text"] = "Writing output"


def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))

    # select folder
    #file_path = filedialog.askdirectory(initialdir="/", title="Select a File")

    # Change label contents
    lbl_value_load.configure(text="File Opened: " + filename)

def plot():

    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),
                 dpi = 100)

    # list of squares
    y = [i**2 for i in range(101)]

    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    plot1.plot(y)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master = window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row=3, column=1)

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().grid(row=4, column=1)


def generate_gui():
    # create a tkinter window
    window = tk.Tk()

    # set 4 rows and 2 columns
    for i in range(5):
        window.rowconfigure(i, minsize=50, weight=1)
        for j in range(2):
            window.columnconfigure(j, minsize=75, weight=1)

    # define load file function
    btn_load_file = tk.Button(master=window, text="Load File", command=browseFiles)
    btn_load_file.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    lbl_value_load = tk.Label(master=window, text="N/A", relief=tk.RAISED)
    lbl_value_load.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    # define solving function
    btn_solve = tk.Button(master=window, text="Solve", command=solve)
    btn_solve.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    lbl_value_solve = tk.Label(master=window, text="N/A", relief=tk.RAISED)
    lbl_value_solve.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    # define writing output file function
    btn_output = tk.Button(master=window, text="Write output", command=output)
    btn_output.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
    lbl_value_output = tk.Label(master=window, text="N/A", relief=tk.RAISED)
    lbl_value_output.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

    # define plot function
    plot_button = tk.Button(master=window, command=plot, text="Plot")
    plot_button.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    # plot_button.pack()

    window.mainloop()
    return window

# if __name__=="__main__":
#     # create a tkinter window
#     window = tk.Tk()
#
#     # set 4 rows and 2 columns
#     for i in range(5):
#         window.rowconfigure(i, minsize=50, weight=1)
#         for j in range(2):
#             window.columnconfigure(j, minsize=75, weight=1)
#
#     # define load file function
#     btn_load_file = tk.Button(master=window, text="Load File", command=browseFiles)
#     btn_load_file.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
#     lbl_value_load = tk.Label(master=window, text="N/A", relief=tk.RAISED)
#     lbl_value_load.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
#
#     # define solving function
#     btn_solve = tk.Button(master=window, text="Solve", command=solve)
#     btn_solve.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
#     lbl_value_solve = tk.Label(master=window, text="N/A", relief=tk.RAISED)
#     lbl_value_solve.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
#
#     # define writing output file function
#     btn_output = tk.Button(master=window, text="Write output", command=output)
#     btn_output.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
#     lbl_value_output = tk.Label(master=window, text="N/A", relief=tk.RAISED)
#     lbl_value_output.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
#
#     # define plot function
#     plot_button = tk.Button(master=window, command=plot, text="Plot")
#     plot_button.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
#     #plot_button.pack()
#
#     window.mainloop()


