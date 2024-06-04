from obs import ObsClient
from obs import GetObjectHeader
import os
import traceback

# 推荐通过环境变量获取AKSK，这里也可以使用其他外部引入方式传入，如果使用硬编码可能会存在泄露风险。
# 您可以登录访问管理控制台获取访问密钥AK/SK，获取方式请参见https://support.huaweicloud.com/intl/zh-cn/usermanual-ca/ca_01_0003.html。
ak ='ZGCNBIRERUNYFPZY1JEW'
sk = 'OzM4hTiyFsOEMZiowqnoIvG2NRF8gUsAZhd4VemX'
# server填写Bucket对应的Endpoint, 这里以中国-香港为例，其他地区请按实际情况填写。
server = 'https://obs.cn-north-4.myhuaweicloud.com'
# 创建obsClient实例
# 如果使用临时AKSK和SecurityToken访问OBS，需要在创建实例时通过security_token参数指定securityToken值
obsClient = ObsClient(access_key_id=ak, secret_access_key=sk, server=server)
# 使用访问OBS

try:
    # 下载对象的附加头域
    headers = GetObjectHeader()

    bucketName = "ioteeth"
    # 指定列举对象的前缀
    prefix = 'obs://ioteeth/position1'
    # 指定单次列举对象个数
    max_keys = 1000
    # 列举桶内对象
    resp = obsClient.listObjects(bucketName, prefix, max_keys=max_keys, encoding_type='url')

    # 下载到本地的路径,localfile为包含本地文件名称的全路径
    downloadPath = 'position1/'

    # 返回码为2xx时，接口调用成功，否则接口调用失败
    if resp.status < 300:
        print('List Objects Succeeded\n')
        #print('requestId:', resp.requestId)
        #print('name:', resp.body.name)
        #print('prefix:', resp.body.prefix)
        #print('max_keys:', resp.body.max_keys)
        #print('is_truncated:', resp.body.is_truncated)
        index = 1
        for content in resp.body.contents:
            print('object [' + str(index) + ']')
            print('地址:', content.key)

            resp2 = obsClient.getObject(bucketName, content.key, downloadPath+str(index)+'.jpg', headers=headers)
            if resp2.status < 300:
                print('下载成功')
                #print('requestId:', resp.requestId)
                #print('url:', resp.body.url)
            else:
                print('Get Object Failed')
                #print('requestId:', resp.requestId)
                #print('errorCode:', resp.errorCode)
                #print('errorMessage:', resp.errorMessage)
            #print('lastModified:', content.lastModified)
            #print('etag:', content.etag)
            #print('size:', content.size)
            #print('storageClass:', content.storageClass)
            #print('owner_id:', content.owner.owner_id)
            #print('owner_name:', content.owner.owner_name)
            index += 1
    else:
        print('List Objects Failed')
        print('requestId:', resp.requestId)
        print('errorCode:', resp.errorCode)
        print('errorMessage:', resp.errorMessage)
except:
    print('List Objects Failed')
    print(traceback.format_exc())