数据表输出成word指定格式
=====
"""
1、获取数据库中表结构
2、每张表增加序列数
3、获取每张表表名作为抬头
4、输出成word表格指定形式
"""

使用规则
    （1）目前只支持MySQL数据库
    （2）配置：
        conf/db.ini  存放数据库连接信息（根据需求更改）
        conf/conf.ini 存放需要输出的数据库