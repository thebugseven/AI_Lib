# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    setup.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jhue <jhue@student.42lyon.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/04/14 14:49:46 by jhue              #+#    #+#              #
#    Updated: 2026/04/14 14:51:02 by jhue             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from setuptools import setup, find_packages

setup(
    name='AI-Toolbox',
    version='1.0.0',
    packages=find_packages(),
    description="A collection of AI tools and utilities"
    " for various applications.",
    author='Jhue',
    author_email='jhue@student.42lyon.fr',
    url='https://github.com/thebugseven/AI_Lib',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Licences :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
    ],
)