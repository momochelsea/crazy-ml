# '0-9' 48-57 => -48 => 0-9
# 'A-Z' 65-90 => -55 => 10-35
# 'a-z' 97-122 => -61 => 36-61


def num2char(num) -> str:
    if num >= 0 and num <= 9:
        num += 48
        return chr(num)
    if num >= 10 and num <= 35:
        num += 55
        return chr(num)
    if num >= 36 and num <= 61:
        num += 61
        return chr(num)
    return ""


def char2num(char) -> int:
    num = ord(char)
    # print(char, num)

    if num >= 48 and num <= 57:
        num -= 48
        return num
    if num >= 65 and num <= 90:
        num -= 55
        return num
    if num >= 97 and num <= 122:
        num -= 61
        return num
    return -1
