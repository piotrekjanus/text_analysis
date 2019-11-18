import codecs

words_with_dot = ['m.in.', 'inż.', 'prof.', 'tzn.', 'np.', 'cd.', 'al.', 'cnd.', 
                  'itp.', 'itd.', 'lek.', 'lic.', 'pl.', 'p.o.', 'św.', 'tj.', 
                  'tzw.', 'ul.', 'zob.', 'ul.']

punctuation = ['.', ':', '(', ')', '?', '!']


def stringify(path : str):
    f = codecs.open(path, "r", encoding = 'utf-8')
    l = list()
    for line in f:
        l.append(line)
    f.close()
    return l[0]


def reverse(text : str):
    '''
    Returns reversed text.
    '''
    return(text[::-1])

def last_n(_list, n):
    '''
    Returns last n elements of list. 
    Returns full list if n is greater than list length or empty string if list is empty
    '''
    if not _list:
        return('')
    if _list[len(_list) - 1] == '':
        _list.pop()
    return(_list[-n:])

def first_n(_list, n):
    '''
    Returns first n elements of list. 
    Returns full list if n is greater than list length or empty string if list is empty
    '''
    if not _list:
        return('')
    if _list[0] == '':
        _list.pop(0)
    return(_list[:n])


def find_annotations(document : str):
    '''
    Searches for all occurances of '<' and '>' in the document.
    Returns lists of indexes of occurances opening for '<' and closing for '>'
    '''
    i = 0
    opening = list()
    closing = list()
    while i != -1:
        i = document.find('<', i)
        opening.append(i)
        if i == -1:
            closing.append(-1)
            break
        i = document.find('>', i)
        closing.append(i)
    closing = [cl + 1 for cl in closing]
    closing[-1] = -1
    return(opening, closing)

def get_annotation_values(text : str): # jest typ nie type, bo type jest zarezerwowana nazwa, nie jestem uposledzony
    '''
    Returns a dict consisting annotation values {'name', 'typ', 'category'} for first occuring annotation
    in the text.
    '''
    name_start = text.find('name=') + len('name=\"')
    name_end = text.find('\"', name_start)
    typ_start = text.find('type=', name_end) + len('type=\"')
    typ_end = text.find('\"', typ_start)
    category_start = text.find('category', typ_end) + len('category=\"')
    category_end = text.find('\"', category_start)
    return({'name' : text[name_start:name_end], 'typ' : text[typ_start:typ_end],
               'category' : text[category_start:category_end]})



def split_the_word(word = '"Pas.chanacz:lolo)u(marek,pies!?"'):
    '''
    Splits the word with elements in punctuation list. Words in words_with_dot are excluded from splitting.
    Returns a list of splitted word. Splitting characters are included in the list.
    '''
    global words_with_dot
    global punctuation
    if word in words_with_dot:
        return([word])
    else:
        l = list()
        index_start = 0
        index_end = 0
        for i, char in enumerate(word):
            if char in punctuation:
                if index_start != index_end:
                    l.append(word[index_start:index_end])
                l.append(char)
                index_start = i + 1
                index_end = i + 1
            else:
                index_end = i + 1
        if index_start != index_end:
            l.append(word[index_start:index_end])
        return(l)
    
def flat_list(_list): # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    '''
    Create one list from list of lists.
    '''
    flat_list = []
    for sublist in _list:
        for item in sublist:
            flat_list.append(item)
    return(flat_list)
    
def repair_sentence(_list, nsize, left):
    '''
    Naprawia zdanie xD. Chodzi o to, zeby oddzielic znaki interpunkcyjne.
    '''
    global words_with_dot
    l = list()
    for el in _list:
        if el not in words_with_dot:
            l.append(split_the_word(el))
        else:
            l.append([el])
    if left:
        return(last_n(flat_list(l), nsize))
    else:
        return(first_n(flat_list(l), nsize))


people_dict = {}

def exclude_vectors_nsize(text, nsize = 3):
    '''
    Parameters:
    text - document string
    nsize - size of window around person word 
    Returns a list of lists build as follows [k_words before person, person, k_words after person], person name, person profession]
    '''
    global people_dict
    opn, cls = find_annotations(text)
    ind = 0
    l = list()
    for i in range(0, len(opn) - 1, 2):
        left_sentence = last_n(text[ind:opn[i]].split(' '), nsize)
        left_sentence = repair_sentence(left_sentence, nsize, left = True)
        right_sentence = first_n(text[cls[i+1]:text.find('<', cls[i+1])].split(' ') , nsize)
        right_sentence = repair_sentence(right_sentence, nsize, left = False)
        annotation = get_annotation_values(text[ind:-1])
        l.append([flat_list([left_sentence, right_sentence]), annotation["name"], annotation["category"]])
        try:
            people_dict[annotation["category"]].add(annotation["name"])
        except KeyError:
            people_dict[annotation["category"]] = {annotation["name"]}
        ind = cls[i+1]
    return l

def exclude_vectors_for_person(list_of_vectors, person):
    '''
    list_of_vectors - return value from exclude_vectors_nsize()
    person - name of person from people_dict
    '''
    l = list()
    for el in list_of_vectors:
        if el[1] == person:
            l.append(el[0])
    return l