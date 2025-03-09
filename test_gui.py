import customtkinter as ctk

def run_test_gui():
    root = ctk.CTk()
    root.geometry("400x300")
    root.title("Test GUI")
    
    label = ctk.CTkLabel(root, text="Fungerar GUI:t?")
    label.pack(pady=20)

    button = ctk.CTkButton(root, text="Stäng", command=root.quit)
    button.pack(pady=20)

    root.mainloop()

run_test_gui()
