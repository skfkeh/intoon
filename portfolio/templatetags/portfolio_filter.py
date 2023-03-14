from django import template
import ast

register = template.Library()

@register.filter
def sub(value, arg):
    return value - arg

@register.filter(name="test")
def str_to_list(value):
    print("----------------template filter-------------------")
    # ast.literal_eval(value)
    return ast.literal_eval(value)
