from flask import Flask, render_template

app = Flask(__name__)   #__name__ 表示flask使用当前模块所在的位置作为程序的根目录


@app.route('/')
def hello_world():  # put application's code here
    #返回给前端的数据
    return '大大大大大大 !'

#添加一个路由和视图函数
@app.route('/index/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    #启动flask服务
    app.run(debug=True,host='0.0.0.0',port=5000)
