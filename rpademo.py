#print("hello world")
import pyautogui
import time

""""
mouse operation
time.sleep(2)
pyautogui.rightClick(100,100)

time.sleep(10)
x,y=pyautogui.position()
print(f'x :{x}, y: {y}')

pyautogui.click(1314,482)

#another operation



pyautogui.drag(100,100, 200,200)
pyautogui.scroll(500)

"""
# Keyboard Operation
"""
time.sleep(2)
pyautogui.write("python rpademo.py")
pyautogui.press('enter')

time.sleep(2)
pyautogui.hotkey("ctrl",'a')
"""

print(pyautogui.size())
ss=pyautogui.screenshot()
ss.save("demo.png")
