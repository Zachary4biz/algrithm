# encoding=utf-8

import os


# cmd+opt+esc 调出强制退出窗口
command1 = """\'tell application \"System Events\"
    key code 53 using {command down, option down}
end tell\'
"""
os.system("osascript -e %s" % command1)

# ctr+left 左屏 (失败)
command2 = """\'tell application \"System Events\"
    key code 126 using (control down)
end tell\'
"""
os.system("osascript -e %s" % command2)

# finder 弹出对话框"Hello World"
command3 = """\'tell application \"Finder\"
    display dialog \"Hello World\"
end tell\'
"""
os.system("osascript -e %s" % command3)


'''
0 0x00 ANSI_A
1 0x01 ANSI_S
2 0x02 ANSI_D
3 0x03 ANSI_F
4 0x04 ANSI_H
5 0x05 ANSI_G
6 0x06 ANSI_Z
7 0x07 ANSI_X
8 0x08 ANSI_C
9 0x09 ANSI_V
10 0x0A ISO_Section
11 0x0B ANSI_B
12 0x0C ANSI_Q
13 0x0D ANSI_W
14 0x0E ANSI_E
15 0x0F ANSI_R
16 0x10 ANSI_Y
17 0x11 ANSI_T
18 0x12 ANSI_1
19 0x13 ANSI_2
20 0x14 ANSI_3
21 0x15 ANSI_4
22 0x16 ANSI_6
23 0x17 ANSI_5
24 0x18 ANSI_Equal
25 0x19 ANSI_9
26 0x1A ANSI_7
27 0x1B ANSI_Minus
28 0x1C ANSI_8
29 0x1D ANSI_0
30 0x1E ANSI_RightBracket
31 0x1F ANSI_O
32 0x20 ANSI_U
33 0x21 ANSI_LeftBracket
34 0x22 ANSI_I
35 0x23 ANSI_P
36 0x24 Return
37 0x25 ANSI_L
38 0x26 ANSI_J
39 0x27 ANSI_Quote
40 0x28 ANSI_K
41 0x29 ANSI_Semicolon
42 0x2A ANSI_Backslash
43 0x2B ANSI_Comma
44 0x2C ANSI_Slash
45 0x2D ANSI_N
46 0x2E ANSI_M
47 0x2F ANSI_Period
48 0x30 Tab
49 0x31 Space
50 0x32 ANSI_Grave
51 0x33 Delete
53 0x35 Escape
55 0x37 Command
56 0x38 Shift
57 0x39 CapsLock
58 0x3A Option
59 0x3B Control
60 0x3C RightShift
61 0x3D RightOption
62 0x3E RightControl
63 0x3F Function
64 0x40 F17
65 0x41 ANSI_KeypadDecimal
67 0x43 ANSI_KeypadMultiply
69 0x45 ANSI_KeypadPlus
71 0x47 ANSI_KeypadClear
72 0x48 VolumeUp
73 0x49 VolumeDown
74 0x4A Mute
75 0x4B ANSI_KeypadDivide
76 0x4C ANSI_KeypadEnter
78 0x4E ANSI_KeypadMinus
79 0x4F F18
80 0x50 F19
81 0x51 ANSI_KeypadEquals
82 0x52 ANSI_Keypad0
83 0x53 ANSI_Keypad1
84 0x54 ANSI_Keypad2
85 0x55 ANSI_Keypad3
86 0x56 ANSI_Keypad4
87 0x57 ANSI_Keypad5
88 0x58 ANSI_Keypad6
89 0x59 ANSI_Keypad7
90 0x5A F20
91 0x5B ANSI_Keypad8
92 0x5C ANSI_Keypad9
93 0x5D JIS_Yen
94 0x5E JIS_Underscore
95 0x5F JIS_KeypadComma
96 0x60 F5
97 0x61 F6
98 0x62 F7
99 0x63 F3
100 0x64 F8
101 0x65 F9
102 0x66 JIS_Eisu
103 0x67 F11
104 0x68 JIS_Kana
105 0x69 F13
106 0x6A F16
107 0x6B F14
109 0x6D F10
111 0x6F F12
113 0x71 F15
114 0x72 Help
115 0x73 Home
116 0x74 PageUp
117 0x75 ForwardDelete
118 0x76 F4
119 0x77 End
120 0x78 F2
121 0x79 PageDown
122 0x7A F1
123 0x7B LeftArrow
124 0x7C RightArrow
125 0x7D DownArrow
126 0x7E UpArrow
'''
