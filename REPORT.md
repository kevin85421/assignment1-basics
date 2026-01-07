# CS336 Language Modeling from Scratch: Assignment 1

## Problem (unicode1): Understanding Unicode (1 point)

`ord()` 把一個 unicode character 轉成 int unicode，`chr()` 把 int unicode 轉成對應的 character。

* (a) What Unicode character does chr(0) return?
    ```sh
    >>> chr(0)
    '\x00'
    ```

* (b) How does this character’s string representation (__repr__()) differ from its printed representa-
tion?
    ```sh
    >>> chr(0).__repr__()
    "'\\x00'"
    ```

* (c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
    ```sh
    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    >>> "this is a test" + chr(0) + "string"
    'this is a test\x00string'
    >>> print("this is a test" + chr(0) + "string")
    this is a teststring
    ```

