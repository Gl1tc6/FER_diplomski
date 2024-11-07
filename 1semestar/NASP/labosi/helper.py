def remove(self,v):
        if self.root is None: return
        n=self.root.query(v)
        fp=n.P
        c=n.children()
        if c==0 and n.P is None: self.root=None
        elif c==0 and n.P is not None:
            if n.P.L is n: n.P.setLeftChild(None)
            elif n.P.R is n: n.P.setRightChild(None)
        elif c==1 and n.P is None:
            if n.L is not None: self.root=n.L.toRoot()
            elif n.R is not None: self.root=n.R.toRoot()
        elif c==1 and n.P is not None:
            if n.P.L is n:
                if n.L is not None: n.P.setLeftChild(n.L)
                elif n.R is not None: n.P.setLeftChild(n.R)
            elif n.P.R is n:
                if n.L is not None: n.P.setRightChild(n.L)
                elif n.R is not None: n.P.setRightChild(n.R)
        elif c==2: # by copy
            pred=n.L.rightmost()
            val=pred.S
            fp=self.remove(pred.S)
            n.S=val
        return fp
    
def query(self,v):
    if self.root is None: return None
    else:
        n=self.root.query(v)
        if v==n.S: return n
        else: return None