from graphviz import Digraph


class Goal:
    def __init__(self, value: str, type: str, father=None, children=None, children_relation='and', context=None,
                 so_that='', having_being_merged=False):
        if children is None:
            children = []
        self.value = value
        self.type = type
        self.father = father
        self.children = children
        self.children_relation = children_relation
        self.context = context
        self.so_that = so_that
        self.having_being_merged = having_being_merged

    def add_child(self, child):
        self.children.append(child)
        child.father = self

    def set_children_relation(self, relation):
        self.children_relation = relation

    def __str__(self):
        children_str = ""
        for child in self.children:
            children_str += str(child) + ", "
        return f"Value: {self.value}, Type: {self.type}, Children: [{children_str[:-2]}], Children Relation: {self.children_relation} "


class Context:
    def __init__(self, value: str, type: str, father=None, children=None, children_relation='and'):
        if children is None:
            children = []
        self.value = value
        self.type = type
        self.father = father
        self.children = children
        self.children_relation = children_relation

    def add_child(self, child):
        self.children.append(child)
        child.father = self

    def set_children_relation(self, relation):
        self.children_relation = relation

    def __str__(self):
        children_str = ""
        for child in self.children:
            children_str += str(child) + ", "
        return f"Value: {self.value}, Type: {self.type}, Children: [{children_str[:-2]}], Children Relation: {self.children_relation}"


class Cnt:
    def __init__(self, cnt=0):
        self.cnt = cnt

    def add(self):
        self.cnt += 1

    def get(self):
        return self.cnt


def draw_goal_model(goal: Goal, dot: Digraph, cnt=Cnt()):
    cnt.add()
    context_dot = Digraph(comment=str(id(goal)) + ' Context Model')
    draw_context_model(goal.context, context_dot, cnt)
    context_dot.render('./result/' + str(id(goal.context)) + 'context_model.png', view=False)
    dot.node(str(id(goal)), goal.value + '\n' + goal.type)
    for child in goal.children:
        dot.node(str(id(child)), child.value + '\n' + child.type + '\n' + str(id(child.context)))
        dot.edge(str(id(goal)), str(id(child)))
        draw_goal_model(child, dot)
    print(cnt.get())


def draw_context_model(context: Context, dot: Digraph, cnt):
    if context is not None:
        cnt.add()
        dot.node(str(id(context)), context.value)
        for child in context.children:
            dot.node(str(id(child)), child.value)
            dot.edge(str(id(context)), str(id(child)))
            draw_context_model(child, dot, cnt)


if __name__ == '__main__':
    # Goal Model
    goal = Goal('Goal', 'goal')
    goal.add_child(Goal('Goal 1', 'goal', goal))
    goal.add_child(Goal('Goal 2', 'goal', goal))
    goal.add_child(Goal('Goal 3', 'goal', goal))
    goal.children[0].add_child(Goal('Goal 1.1', 'goal', goal.children[0]))
    goal.children[0].add_child(Goal('Goal 1.2', 'goal', goal.children[0]))
    goal.children[0].children[0].add_child(Goal('Goal 1.1.1', 'goal', goal.children[0].children[0]))
    goal.children[0].children[0].add_child(Goal('Goal 1.1.2', 'goal', goal.children[0].children[0]))
    goal.children[0].children[0].children[0].add_child(
        Goal('Goal 1.1.1.1', 'goal', goal.children[0].children[0].children[0]))
    dot = Digraph(comment='Goal Model')
    draw_goal_model(goal, dot)
    dot.render('goal_model.png', view=True)

    # Context Model
    context = Context('Context', 'context')
    context.add_child(Context('Context 1', 'context', context))
    context.add_child(Context('Context 2', 'context', context))
    context.add_child(Context('Context 3', 'context', context))
    context.children[0].add_child(Context('Context 1.1', 'context', context.children[0]))
    context.children[0].add_child(Context('Context 1.2', 'context', context.children[0]))
    context.children[0].children[0].add_child(Context('Context 1.1.1', 'context', context.children[0].children[0]))
    context.children[0].children[0].add_child(Context('Context 1.1.2', 'context', context.children[0].children[0]))
    context.children[0].children[0].children[0].add_child(
        Context('Context 1.1.1.1', 'context', context.children[0].children[0].children[0]))
    dot = Digraph(comment='Context Model')
    draw_context_model(context, dot)
    dot.render('context_model.png', view=True)
