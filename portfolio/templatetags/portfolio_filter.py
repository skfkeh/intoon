from django import template
import ast

register = template.Library()

@register.filter
def sub(value, arg):
    return value - arg

@register.filter(name="test")
def str_to_list(value):
    # ast.literal_eval(value)
    return ast.literal_eval(value)

@register.filter(name="slicing_img")
def slicing_img(value):
    return value.slice(',')[0][2:]