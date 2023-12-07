
def specialize(text):
    return text.replace("<<<","").replace(">>>","").replace("<<</","").replace("/>>>","")

class TreeNode:
    def __init__(self, index: str, name: str, content: str):
        self.index = index if index != "-1" else ""
        self.name = specialize(name)
        self.content = specialize(content)
        self.children = []

    def add_child(self, child):
        if child:
            self.children.append(child)
    
    def __str__(self) -> str:
        return "{} {}: {}.".format(self.index, self.name, self.content)
    
    def __repr__(self) -> str:
        if len(self.children) == 0:
            return "<<<{} {}>>> {} <<</{} {}/>>>".format(self.index, self.name, self.content, self.index, self.name)
        else:
            child = [repr(ch) for ch in self.children]
            return "<<<{} {}>>> {} {} <<</{} {}/>>>".format(self.index, self.name, self.content, " ".join(child), self.index, self.name)

    def pre_traverse(self) -> str:
        prompt = self.__str__() + "\n"
        for child in self.children:
            prompt += str(child) + "\n"
        return prompt

    def deep_traverse(self) -> list:
        if len(self.children) == 0:
            prompts = ["<<<{} {}>>> {} <<</{} {}/>>>".format(self.index, self.name, self.content, self.index, self.name)]
        else:
            prompts = []
            temp = "<<<{} {}>>> {} {} <<</{} {}/>>>"
            for child in self.children:
                for ch in child.deep_traverse():
                    prompts.append(temp.format(self.index, self.name, self.content, ch, self.index, self.name))
        return prompts

def build_tree(body):
    if not body:
        return None

    data = body.pop(0)
    index = str(data["section"]["index"]).strip()
    name = data["section"]["name"]
    content = "\n".join(data["p"])
    Node = TreeNode(index, name, content)
    if "-1" in index:
        return Node
    
    while body:
        index_next = str(body[0]["section"]["index"]).strip()
        if "-1" in index_next or index_next.find(index) == 0:
            child = build_tree(body)
            Node.add_child(child)
        else:
            break
    
    return Node
