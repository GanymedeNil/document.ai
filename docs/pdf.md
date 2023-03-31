# pdf 文件解析

pdf 格式目前是最常见的文档格式，但是它的解析并不是很容易，因为它是一个二进制文件，而且它的结构非常复杂，所以解析起来比较困难。

calibre（免费电子书管理软件）也吐槽了pdf的解析难度：[convert pdf documents](https://manual.calibre-ebook.com/conversion.html#convert-pdf-documents)

以我目前的经验来看，pdf 的解析主要分三种，并给出了对应的开源解决方案：

# 纯文本和图片

这种纯文本的解析相对容易，以下几个库即可进行解析：

python：[pypdf](https://github.com/py-pdf/pypdf)

java：[pdfbox](https://pdfbox.apache.org/)

Go：[pdfcpu](https://github.com/pdfcpu/pdfcpu)

# 学术类pdf（公式和表格）

这种解析比较困难，因为它的结构比较复杂

我目前尝试比较好的是使用 [grobid](https://github.com/kermitt2/grobid)，它是一个开源的学术文档解析库，先通过它转为 TEI
的xml格式，这样就可以很方便的进行解析了，并且也提供了表格和公式的坐标，可以截取后通过相关OCR来进行识别。

# 纯图片

全部都为图片需要将图片转为文本才能进行后续的动作

传统OCR：

[OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF)

基于VDU模型:

[donut](https://github.com/clovaai/donut)
