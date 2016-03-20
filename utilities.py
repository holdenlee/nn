import itertools

def case(var, li, else_expr=None):
    for (val, expr) in li:
        if val:
            return expr
    return else_expr

#alternatively use the ternary control operator
#    a if test else b
#http://stackoverflow.com/questions/394809/does-python-have-a-ternary-conditional-operator
#WARNING: THIS IS NOT LAZY
def if_f(expr, t, f):
    if expr:
        return t
    else:
        return f

def ifs(li, else_expr=None):
    for (stmt, val) in li:
        if stmt:
            return val
    return else_expr

def concat(lis):
    return itertools.chain(*lis)

def union(*dicts):
    return dict(sum(map(lambda dct: list(dct.items()), dicts), []))
