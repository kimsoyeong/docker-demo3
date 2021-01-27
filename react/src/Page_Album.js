import React, {useEffect} from 'react'
import NavigationBar from './NavigationBar'
import './index.css';
import Gallery from 'react-grid-gallery';
import axios from "axios";

const IMAGES =
[{
        src: "https://c2.staticflickr.com/9/8817/28973449265_07e3aa5d2e_b.jpg",
        thumbnail: "https://c2.staticflickr.com/9/8817/28973449265_07e3aa5d2e_n.jpg",
        thumbnailWidth: 320,
        thumbnailHeight: 174,
        isSelected: true,
        caption: "After Rain (Jeshu John - designerspics.com)"
},
{
        src: "https://c2.staticflickr.com/9/8356/28897120681_3b2c0f43e0_b.jpg",
        thumbnail: "https://c2.staticflickr.com/9/8356/28897120681_3b2c0f43e0_n.jpg",
        thumbnailWidth: 320,
        thumbnailHeight: 212,
        tags: [{value: "Ocean", title: "Ocean"}, {value: "People", title: "People"}],
        caption: "Boats (Jeshu John - designerspics.com)"
},

{
        src: "https://c4.staticflickr.com/9/8887/28897124891_98c4fdd82b_b.jpg",
        thumbnail: "https://c4.staticflickr.com/9/8887/28897124891_98c4fdd82b_n.jpg",
        thumbnailWidth: 320,
        thumbnailHeight: 212
}]

function Page_UploadImage({history}) {
    useEffect(() => {  
        axios.get("/get_album")
        .then((Response) => {
            console.log(Response.data);
        }).catch((Error) => {
            console.log(Error);
        })
    }, [])
        
    return (
        <NavigationBar history={history} icon={"camera"} pageName={"ALBUM"} content={
            <div className="App-container">
                <Gallery images={IMAGES}/>
            </div>
        }/>
    )
}

export default Page_UploadImage;
