"""
Configuration 파일을 인자로 받음.

Blocks 리스트를 리턴함.
각 block은 neural network를 어떻게 빌드하는지에 대해 나타냄.
Blocks는 dictionary들의 리스트임.

함수의 목적 : cfg를 파싱 => 모든 block을 dict 형식으로 저장하는 것.

Block들의 attribute들과 value들은 dictionary에 key-value 형식으로 저장됨.

code : cfg 파싱 => 이러한 dicts들을 block 변수에 저장 => blocks라는 list에 dicts들을 append.

return : list

"""

def parse_cfg(cfgfile) :

    file = open(cfgfile, 'r') # file open
    lines = file.read().split('\n') # 개행(line) 기준으로 나누어 list로 저장
    lines = [line for line in lines if len(line) > 0] # 비어있지 않은 line들만 추출 (빈 line 제거)
    lines = [line for line in lines if line[0] != '#'] # 주석이 아닌 line들만 추출 (주석 line 제거)
    lines = [line.rstrip().lstrip() for line in lines] # white space 제거

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":      # new block start
            if len(block) != 0:     # block is not empty => append it next the previous block
                blocks.append(block)        # append to block list
                block = {}      # empty block
            block["type"] = line[1:-1].rstrip()         # 대괄호를 제외한 block의 이름을 dict로 생성
            # ie : block = {"type" : 'net'}
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
            # ie : block = {'batch' : 64, 'subdivisions'=16, ...}
    blocks.append(block)
    # ie : blocks = [{"type" : 'net', 'batch' : 64, ...}, {"type" : 'convolution', 'filters' : 32, ...}]

    return blocks