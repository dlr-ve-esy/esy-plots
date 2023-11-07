import uuid
from random import randint
import streamlit as st
# from header import load_header


def create_layout(elements):

    for i, element in enumerate(elements):
        simple_layout(key=i, **element)


def simple_layout(
        radio_label: str, radio_options: list[str],
        plotting_function: object, radio_kwargs: dict=None,
        key: int=0
    ):

    radio_kwargs = radio_kwargs if radio_kwargs is not None else {}
    option = st.radio(
        label=radio_label,
        options=radio_options,
        key=key,
        **radio_kwargs
    )
    figure = plotting_function(option)
    st.bokeh_chart(figure, use_container_width=True)
