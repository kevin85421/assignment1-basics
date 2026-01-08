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

## Problem (unicode2): Unicode Encodings (3 points)

Unicode 不適合用來訓練 tokenizer，因為這樣 vocabulary 會變太多 (~150K)，而且很多 character 很少用到。
因此將 Unicode string 進行 encode 轉換成 `list[bytes]` (UTF-8)。

* `utf8_encoded = s.encode("utf-8")` => 把 `s` 轉換成 UTF-8 bytes。
* `utf8_encoded.decode("utf-8")` => decode 後又變為 `s`。 

* (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
  * 使用 UTF-32、UTF-16 encode 的結果會比 UTF-8 更長 (更多 bytes)，因此訓練 tokenizer 時需要更多記憶體，也越慢。 
    ```sh
    >>> s = "A"
    >>> s.encode("utf-8")
    b'A'
    >>> s.encode("utf-16")
    b'\xff\xfeA\x00'
    >>> s.encode("utf-32")
    b'\xff\xfe\x00\x00A\x00\x00\x00'
    ```

* (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
  * 不是所有 character 進行 encode 後都是一個 byte，如果 encode 後是多個 bytes，那也要多個 bytes 同時進行 decode 才會得到原來的結果。
    ```python
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    s = "你好"
    encoded_s = s.encode("utf-8")
    print(encoded_s)
    print(decode_utf8_bytes_to_str_wrong(encoded_s))
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
    ```

* (c) Give a two byte sequence that does not decode to any Unicode character(s).
    * "你好" 中的 2 個 bytes
        ```python
        >>> b = b'\xe4'
        >>> type(b)
        <class 'bytes'>
        >>> b
        b'\xe4'
        >>> b.decode("utf-8")
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
        >>> b = b'\xe5'
        >>> b.decode("utf-8")
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe5 in position 0: unexpected end of data
        ```
