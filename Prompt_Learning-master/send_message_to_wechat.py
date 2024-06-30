import requests

mydata={
'text':'Expriment Complete.',
'desp':'The expriment on 531_new_single_3060 is complete,come to check the result.'
}

token = 'fZ6Rc0WNLx9q91EDxP1lSF8As' # 记得替换成你自己的token，不然发错消息给别人了

url = "https://wx.xtuis.cn/" + token + ".send"

print(requests.post(url, data=mydata))