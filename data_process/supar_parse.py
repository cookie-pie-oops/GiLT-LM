from supar import Parser
# if the gpu device is available
# >>> torch.cuda.set_device('cuda:0')  
if __name__ == '__main__':
    parser = Parser.load('biaffine-sdp-en')
    # parser = Parser.load('vi-sdp-roberta-en')
    print("load success!")
    dataset = parser.predict([[('I',None,'PRP'), ('saw',None,'VBD'), ('Sarah',None,'NNP'), ('with',None,'IN'),('a',None,'DT'), ('telescope',None,'NN'), ('.',None,'.')],[('I','I','PRP'), ('saw','see','VBD'), ('Sarah','Sarah','NNP'), ('with','with','IN'),('a','a','DT'), ('telescope','telescope','NN'), ('.','_','.')]],verbose=False)
    import pdb;pdb.set_trace()
    dataset[0].values