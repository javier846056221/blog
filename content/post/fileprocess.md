
## 创建目录
import os
> os.makedirs(path, mode, exist_ok)
该函数用来递归创建多层目录

path：path是递归创建的目录，可以是相对路径或者绝对路径
mode：mode是权限模式，默认值是511（八进制）
exist_ok：是否在目录存在时触发异常。如果 exist_ok 为 False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；如果 exist_ok 为 True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。

## 合并文件
> os.path.join(path1[], path2[, ...])
该函数用来将目录和文件名合成一个路径，如：
data_file = os.path.join('..', 'data', 'house_tiny.csv')

`'..'`: 这表示上一级目录，也就是当前工作目录的父目录。如果当前工作目录是`/home/user/documents/`，那么`'..'`将指向`/home/user/`。
这个语句创建了一个文件路径，该路径指向名为'house_tiny.csv'的数据文件，该文件位于当前工作目录的父目录中的"data"子目录内。`os.path.join()` 函数返回完整的文件路径字符串。
