# Shakespear-Generator

This Shakespear generator was built following the transformers paper `Attention Is All You Need`.
https://en.wikipedia.org/wiki/Attention_Is_All_You_Need
https://arxiv.org/abs/1706.03762

In the files, there are the Jupiter notebooks, where I documented some tests and where I learnt the majority of what was used in this project. The model is built, trained,and exported in the model.py file. In generate.py, the model is imported and used to generate Shakespearian style text. You can chose whether or not you want the model to be trained using character tokenization or word tokenization, which is much longer to train. Since I did not have enough ressources, the current model was only trained on a character-level basis, so there is currently no coherent story. 

I did not rely on OpenAI, nor the OpenAI API to make this project possible.

Output example:

LADY CAPULET:
Let thy sword dream? By God, let it gleam!

DUKE OF YORK:
Yes, then, come, my lord: live by my grace and me.
When, how that I move, so be the walk with thee.

QUEEN ELIZABETH:
’Twas pardon—I do not entertain it so.

GLOUCESTER:
I would see this deed so done were in the man:
I would not say it is his nature to lie!
That, I’ll foul mock of my heart—Cousin, aye.

VOLUMNIA:
Brother, my chosen hearth—kill, fellow.

SICINIUS:
Where harmony, a Christening of foul in this dares
Men, meaning in the mercy war fares.
